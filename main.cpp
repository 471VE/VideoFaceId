#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include "MultiTracker.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <chrono>
#include <time.h>
#include <Windows.h>


using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

cv::VideoCapture capture;

const int DICT_SIZE = 50;
const double LOG_REG_THRESHOLD = 0.35;

cv::Mat FaceFeatureVector(const cv::Mat& descriptors, const cv::Mat& k_centers) {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, k_centers, matches);

    cv::Mat feature_vector = cv::Mat::zeros(1, DICT_SIZE, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++) {
        feature_vector.at<float>(0, matches.at(index).trainIdx) = feature_vector.at<float>(0, matches.at(index).trainIdx) + 1;
    }
    return feature_vector;
}

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
            full_frame, names[i], cv::Point(faces[i].x * scale, (faces[i].y + faces[i].height) * scale + 25),
            cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
}

void ComputeClassesProbabilities(cv::Mat& logit_mat) {
    float sum = 0;
    for (int i = 0; i < logit_mat.cols; ++i) {
        logit_mat.at<float>(0, i) = exp(logit_mat.at<float>(0, i));
        sum += logit_mat.at<float>(0, i);
    }
    for (int i = 0; i < logit_mat.cols; ++i) {
        logit_mat.at<float>(0, i) /= sum;
    }        
}

void FaceRecognition(
    std::string filename,
    const std::vector<std::string>& names,
    cv::Ptr<cv::ml::LogisticRegression>& classifier,
    const cv::Mat& k_centers,
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
    std::vector<std::string> names_of_detected_faces;

    bool tracked;

    while (true) {
        capture >> full_frame;
        if (full_frame.empty())
            break;
        frame_count++;
        cv::resize(full_frame, frame_downscaled, cv::Size(), scale_inverse, scale_inverse);
        cvtColor(frame_downscaled, frame_gray, cv::COLOR_BGR2GRAY);

        tracked = false;
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
                } else
                    tracked = true;
            }
        }
        else
            face_cascade.detectMultiScale(frame_downscaled, faces);

        if (!tracked) {
            names_of_detected_faces.clear();
            for (const auto& face: faces) {
                detector->detectAndCompute(
                    full_frame(cv::Rect(face.x*scale, face.y*scale, face.width*scale, face.height*scale)),
                    cv::Mat(), person_keypoints_tmp, person_descriptors);
                cv::Mat feature_vector = FaceFeatureVector(person_descriptors, k_centers);

                cv::Mat feature_vector_added;
                cv::hconcat(cv::Mat::ones(1, 1, CV_32F), feature_vector, feature_vector_added);
                cv::Mat product = (classifier->get_learnt_thetas() * feature_vector_added.t()).t();
                ComputeClassesProbabilities(product);

                std::vector<float> probabilities;
                probabilities.assign((float*)product.data, (float*)product.data + product.total()*product.channels());
                std::vector<std::pair<float, size_t>> ordered_probabilities;
                for (size_t i = 0; i < probabilities.size(); ++i) {
                    ordered_probabilities.push_back(std::make_pair(probabilities[i], i));
                }

                std::sort(ordered_probabilities.begin(), ordered_probabilities.end(), std::greater<>());

                if (ordered_probabilities[0].first < ordered_probabilities[1].first + 0.02)
                    names_of_detected_faces.push_back("Unknown");
                else
                    names_of_detected_faces.push_back(names[ordered_probabilities[0].second]);
            }
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

void LoadDatasetSIFT(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Mat& all_descriptors, std::vector<cv::Mat>& all_descriptors_by_image,
    std::vector<int>& all_classes_labels, int& all_images_num)
{
    all_images_num = 0;
    for (int i = 0; i < names.size(); ++i) {
        auto path_to_person_descriptors = dataset_path + "\\" + names[i] + "\\descriptors\\SIFT";

        for(auto& descriptor_path: std::filesystem::directory_iterator(path_to_person_descriptors)) {
            auto tmp = cv::imread(descriptor_path.path().string(), cv::IMREAD_GRAYSCALE);
            cv::Mat_<float> tmp_float;
            tmp.convertTo(tmp_float, CV_32F);

            all_descriptors.push_back(tmp_float);
            all_descriptors_by_image.push_back(tmp_float);
            all_classes_labels.push_back(i);
            all_images_num++;
        }
    }
}


void TrainPersonClassifier(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Ptr<cv::ml::LogisticRegression>& classifier, cv::Mat& k_centers)
{
    cv::Mat all_descriptors;
    std::vector<cv::Mat> all_descriptors_by_image;
    std::vector<int> all_classes_labels;
    int all_images_num;

    LoadDatasetSIFT(names, dataset_path, all_descriptors, all_descriptors_by_image, all_classes_labels, all_images_num);

    int cluster_count = DICT_SIZE;
    int attempts = 5;
    int iteration_number = static_cast<int>(1e4);

    cv::Mat k_labels;
    kmeans(
        all_descriptors, cluster_count, k_labels,
        cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, iteration_number, 1e-4),
        attempts, cv::KMEANS_PP_CENTERS, k_centers);

    cv::Mat input_data;
    cv::Mat input_data_labels;

    for (int i = 0; i < all_images_num; i++) {
        cv::BFMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(all_descriptors_by_image[i], k_centers, matches);

        cv::Mat feature_vector = cv::Mat::zeros(1, DICT_SIZE, CV_32F);
        int index = 0;
        for (auto j = matches.begin(); j < matches.end(); j++, index++) {
            feature_vector.at<float>(0, matches.at(index).trainIdx) = feature_vector.at<float>(0, matches.at(index).trainIdx) + 1;
        }

        input_data.push_back(feature_vector);
        input_data_labels.push_back(cv::Mat(1, 1, CV_32F, all_classes_labels[i]));
    }

    classifier->setLearningRate(0.001);
    classifier->setIterations(100);
    classifier->setRegularization(cv::ml::LogisticRegression::REG_L2);
    classifier->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
    classifier->setMiniBatchSize(100);

    cv::Ptr<cv::ml::TrainData> training_data = cv::ml::TrainData::create(input_data, cv::ml::ROW_SAMPLE, input_data_labels);
    classifier->train(training_data);
}


int main() {
    std::string dataset_path = "..\\..\\..\\training_set";
    auto names = GetDirectories(dataset_path);

    std::cout << "Training the model...\n";
    cv::Mat k_centers;
    cv::Ptr<cv::ml::LogisticRegression> classifier = cv::ml::LogisticRegression::create();
    TrainPersonClassifier(names, dataset_path, classifier, k_centers);
    std::cout << "Training complete.\n";

    std::string filename = "..\\..\\..\\test\\bateman_interrogation_cut.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename, names, classifier, k_centers, "KCF");
    return 0;
}