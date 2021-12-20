#include <vector>
#include <iostream>
#include <string>

#include "MultiTracker.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <time.h>
#include <Windows.h>


using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

const double kDistanceCoef = 4.0;

cv::VideoCapture capture;

void RealTimeSyncing(
    const Time::time_point& video_start,
    Time::time_point& frame_end,
    cv::VideoCapture& capture,
    double& total_time_actual,
    double& total_time_predicted,
    const double& frame_time,
    cv::Mat& full_frame,
    int& frame_count,
    uchar& wait_time,
    int& keyboard)
{
    frame_end = Time::now();
    total_time_actual = std::chrono::duration_cast<ns>(frame_end - video_start).count() * 1e-6;
    total_time_predicted = frame_time * static_cast<double>(frame_count);
    if (total_time_actual > total_time_predicted) {
        if (total_time_actual > total_time_predicted + frame_time) {
            // Skip frames if program runs too slow
            int excess_frames = static_cast<int>((total_time_actual - total_time_predicted) / frame_time);
            for (int i = 0; i < excess_frames; ++i) {
                capture >> full_frame;
                frame_count++;
            }
        }
        // Immediately go to the next frame if delay is less than one frame or necesary frames have already been skipped
        keyboard = cv::waitKey(1);
    } else {
        // Wait if processing is ahead of audio
        wait_time = cv::saturate_cast<uchar>(total_time_predicted - total_time_actual) + 1;
        keyboard = cv::waitKey(wait_time);
    }
}


void StartAudioPlayback(const std::string& filename, Time::time_point& video_start, bool& first_frame) {
    std::string mci_string = "open " + filename + " type mpegvideo alias AudioFile";
    LPCSTR mci_command = mci_string.c_str();
    mciSendString(mci_command, 0, 0, 0);
    mciSendString("play AudioFile from 0", 0, 0, 0);
    video_start = Time::now();
    mciSendString("window AudioFile state hide", 0, 0, 0);
    first_frame = false;
}


cv::CascadeClassifier face_cascade("../../../haarcascade/haarcascade_frontalface_alt2.xml");

void DrawFaces(
    cv::Mat full_frame,
    const std::vector<cv::Rect>& faces,
    const int& scale,
    const std::vector<std::string>& names)
{
    for (int i = 0; i < faces.size(); ++i) {
        cv::rectangle(
            full_frame, cv::Rect(
                faces[i].x*scale, faces[i].y*scale,
                faces[i].width*scale, faces[i].height*scale),
            cv::Scalar(0, 0, 255), 2, 1);
        cv::putText(
            full_frame, names[i], cv::Point(faces[i].x * scale, faces[i].y * scale - 20),
            cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
}

void match(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& good_matches) {
    good_matches.clear();

    cv::BFMatcher desc_matcher(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> vmatches;
    desc_matcher.knnMatch(desc1, desc2, vmatches, 2);

    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
        if (!vmatches[i].size()) {
            continue;
        }
        if (vmatches[i][0].distance < 0.75 * vmatches[i][1].distance)
            good_matches.push_back(vmatches[i][0]);
    }

    std::sort(good_matches.begin(), good_matches.end());
}


void FaceRecognition(
    std::string filename,
    std::vector<std::vector<cv::Mat>> dataset, const std::vector<std::string>& names,
    const std::string& tracker_type = "NO_TRACKER")
{
    double frame_time = 1000. / capture.get(cv::CAP_PROP_FPS);
    Time::time_point video_start, frame_end;
    uchar wait_time;
    int keyboard = 0;

    cv::Mat full_frame, frame_downscaled, frame_gray;
    int scale = 4;
    double scale_inverse = 1. / static_cast<double>(scale);

    bool first_frame = true;
    int frame_count = 0;
    double total_time_actual, total_time_predicted;

    std::vector<cv::Rect> faces;
    MultiTracker trackers = MultiTracker(tracker_type);

    auto detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> person_keypoints_tmp;
    cv::Mat person_descriptors;
    std::vector<cv::DMatch> good_matches;
    std::vector<std::pair<size_t, std::string>> good_matches_num;
    std::vector<std::string> names_of_detected_faces;

    while (true) {
        capture >> full_frame;
        if (full_frame.empty())
            break;
        frame_count++;

        cv::resize(full_frame, frame_downscaled, cv::Size(), scale_inverse, scale_inverse);
        cvtColor(frame_downscaled, frame_gray, cv::COLOR_BGR2GRAY);

        if (tracker_type != "NO_TRACKER") {
            if (frame_count % 20 == 1 || faces.size() == 0) {
                faces.clear();
                face_cascade.detectMultiScale(frame_gray, faces);
                trackers.start(frame_downscaled, faces);
            }
            else {
                if (!trackers.update(frame_downscaled, faces)) {
                    faces.clear();
                    face_cascade.detectMultiScale(frame_gray, faces);
                    trackers.start(frame_downscaled, faces);
                }
            }
        }
        else
            face_cascade.detectMultiScale(frame_downscaled, faces);

        names_of_detected_faces.clear();
        for (const auto& face: faces) {
            detector->detectAndCompute(
                full_frame(cv::Rect(face.x*scale, face.y*scale, face.width*scale, face.height*scale)),
                cv::Mat(), person_keypoints_tmp, person_descriptors);                
            for (size_t i = 0; i < dataset.size(); ++i) {
                for (const auto& true_descriptors: dataset[i]) {
                    match(person_descriptors, true_descriptors, good_matches);
                    good_matches_num.push_back(std::make_pair(good_matches.size(), names[i]));
                }
            }
            std::sort(good_matches_num.begin(), good_matches_num.end(), std::greater<>());
            names_of_detected_faces.push_back(good_matches_num[0].second);
        }
        
        DrawFaces(full_frame, faces, scale, names_of_detected_faces);
        cv::imshow("Face Detection", full_frame);

        if (first_frame) {
            first_frame = false;
            StartAudioPlayback(filename, video_start, first_frame);
        }

        RealTimeSyncing(
            video_start, frame_end, capture, total_time_actual, total_time_predicted,
            frame_time, full_frame, frame_count, wait_time, keyboard);

        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}

void ExtractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    auto detector = cv::SIFT::create();
    detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}

std::vector<std::string> GetDirectories(const std::string& path_to_dir) {
    std::vector<std::string> directories;
    for(auto& element: std::filesystem::directory_iterator(path_to_dir))
        if (element.is_directory()) {
            std::string full_path = element.path().string();
            std::string short_path = full_path.substr(full_path.find_last_of("\\") + 1, full_path.size());
            directories.push_back(short_path);
        }
    return directories;
}

std::vector<std::vector<cv::Mat>> LoadDatasetSIFT(const std::vector<std::string>& names, const std::string& dataset_path) {
    std::vector<cv::Mat> person_descriptors;
    std::vector<std::vector<cv::Mat>> dataset;
    for (const auto& name: names) {
        person_descriptors.clear();
        auto path_to_person_descriptors = dataset_path + "\\" + name + "\\descriptors\\SIFT";
        for(auto& descriptor_path: std::filesystem::directory_iterator(path_to_person_descriptors)) {
            auto tmp = cv::imread(descriptor_path.path().string(), cv::IMREAD_GRAYSCALE);
            cv::Mat_<float> tmp_float;
            tmp.convertTo(tmp_float, CV_32F);
            person_descriptors.push_back(tmp_float);
        }
        dataset.push_back(person_descriptors);
    }
    return dataset;
}



int main() {
    std::string dataset_path = "..\\..\\..\\training_set";
    auto names = GetDirectories(dataset_path);
    auto dataset = LoadDatasetSIFT(names, dataset_path);
    std::string filename = "..\\..\\..\\test\\ford_gosling.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename, dataset, names);
    return 0;
}