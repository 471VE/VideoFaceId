#ifndef VIDEO_PROCESSING_H
#define VIDEO_PROCESSING_H

#include <vector>
#include <string>

#include <chrono>
#include <time.h>

#include <opencv2/opencv.hpp>

#include "MultiTracker.h"

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

std::vector<std::string> GetNames(const std::string& path_to_dir);
std::vector<std::vector<cv::Mat>> LoadDataset(const std::vector<std::string>& names, const std::string& dataset_path);
void match(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& good_matches);

void StartAudioPlayback(const std::string& filename, Time::time_point& video_start, bool& first_frame);
void StopAudioPlayback(const std::string& filename);

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

void DrawFaces(
    cv::Mat full_frame,
    const std::vector<cv::Rect>& faces,
    const double& scale,
    const std::vector<std::string>& names);

void FaceIdentification(
    std::vector<std::string>& names_of_detected_faces,
    const std::vector<std::string>& names,
    const std::vector<cv::Rect>& faces,
    cv::Ptr<cv::SIFT>& detector,
    const cv::Mat& full_frame,
    const double& scale,
    std::vector<cv::KeyPoint>& person_keypoints_tmp,
    cv::Mat& person_descriptors,    
    std::vector<cv::DMatch>& good_matches,
    std::vector<std::pair<size_t, std::string>>& good_matches_num,
    const std::vector<std::vector<cv::Mat>>& dataset);

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
    cv::VideoCapture& capture,
    const std::vector<std::vector<cv::Mat>>& dataset,
    const std::string& tracker_type = "KCF");

#endif // VIDEO_PROCESSING_H