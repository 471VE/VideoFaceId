#include <vector>
#include <iostream>
#include <string>

#include "MultiTracker.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <time.h>
#include <Windows.h>

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

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

void DrawFaces(cv::Mat full_frame, const std::vector<cv::Rect>& faces, const int& scale) {
    for (const auto& face: faces) {
        cv::rectangle(
            full_frame, cv::Rect(
                face.x*scale, face.y*scale,
                face.width*scale, face.height*scale),
            cv::Scalar(255, 0, 0), 2, 1);
            cv::waitKey(1);
    }
}


void FaceRecognition(std::string filename, const std::string& tracker_type = "NO_TRACKER") {
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

        DrawFaces(full_frame, faces, scale);
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





int main() {
    std::string filename = "../../../test/ford_gosling.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename);
    return 0;
}