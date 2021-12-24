#ifndef VIDEO_PROCESSING_H
#define VIDEO_PROCESSING_H

#include <vector>
#include <string>

#include <chrono>
#include <time.h>

#include <opencv2/opencv.hpp>

#include "MultiTracker.h"

const double LOG_REG_THRESHOLD = 0.35;

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

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
    int& keyboard);

void StartAudioPlayback(const std::string& filename, Time::time_point& video_start, bool& first_frame);
void StopAudioPlayback(const std::string& filename);

void DrawFaces(
    cv::Mat full_frame,
    const std::vector<cv::Rect>& faces,
    const int& scale,
    const std::vector<std::string>& names);

void FaceIdentification(
    std::vector<std::string>& names_of_detected_faces,
    const std::vector<std::string>& names,
    const std::vector<cv::Rect>& faces,
    cv::Ptr<cv::SIFT>& detector,
    const cv::Mat& full_frame,
    const int& scale,
    std::vector<cv::KeyPoint>& person_keypoints_tmp,
    cv::Mat& person_descriptors,
    cv::Mat& feature_vector,
    cv::Mat& feature_vector_with_bias,
    std::vector<float>& probabilities,
    std::vector<std::pair<float, size_t>>& ordered_probabilities,
    const cv::Mat& k_centers,
    const cv::Mat& classifier_params);

void TrackOrDetect(
    const std::string& tracker_type,
    const int& frame_count,
    std::vector<cv::Rect>& faces,
    const cv::Mat& frame_downscaled,
    const cv::Mat& frame_gray,
    MultiTracker& trackers,
    bool& tracked);

void FaceRecognition(
    const std::string& filename,
    const std::vector<std::string>& names,
    const cv::Mat& classifier_params,
    const cv::Mat& k_centers,
    cv::VideoCapture& capture,
    const std::string& tracker_type = "KCF");

#endif // VIDEO_PROCESSING_H