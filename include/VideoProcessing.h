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

// Gets names of people in the database.
std::vector<std::string> GetNames(const std::string& path_to_dir);
// Loads previously created descriptors for all people.
std::vector<std::vector<cv::Mat>> LoadDataset(const std::vector<std::string>& names, const std::string& dataset_path);

// The main function that processes the video from start to finish.
void FaceRecognition(
    const std::string& filename,
    const std::vector<std::string>& names,
    cv::VideoCapture& capture,
    const std::vector<std::vector<cv::Mat>>& dataset,
    double& true_positives,
    double& false_positives,
    double& false_negatives,
    std::vector<std::vector<double>>& classes_statistics,
    const bool& skip_stats,
    const std::string& tracker_type = "KCF");

void PrintDetectionStatistics(const double& true_positives, const double& false_positives, const double& false_negatives);
void PrintClassesStatistics(const std::vector<std::vector<double>>& classes_statistics, const std::vector<std::string>& names);

#endif // VIDEO_PROCESSING_H