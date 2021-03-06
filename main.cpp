#include <vector>
#include <iostream>
#include <string>
#include <filesystem>

#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>

cv::VideoCapture capture;

bool single_video_flag = false;
bool skip_stats = false;
std::string video_name;
std::string video_path;

void ParseCmdArgs(int argc, char** argv) {
    if (argc <= 4) {
        if (argc == 2) {
            if (std::string(argv[1]) != "--skip-stats") {
                std::cerr << "\nWith this number od arguments, you can only specify the \"--skip_stats\" flag ";
                std::cerr << "to skip the calculation of statistics. Exiting...";
                exit(1);
            }
            skip_stats = true;
        }
        if (argc == 3 || argc == 4) {
            if (std::string(argv[1]) != "--video") {
                std::cerr << "\nOnly the name of the video may be specified and/or the \"--skip_stats\" flag, check the command. Exiting...";
                exit(1);
            } else {
                single_video_flag = true;
                video_name = std::string(argv[2]);
            }
            if (argc == 4) {
                if (std::string(argv[3]) != "--skip-stats") {
                    std::cerr << "\nThe last argument may only be the \"--skip_stats\" flag ";
                    std::cerr << "to skip the calculation of statistics. Exiting...";
                    exit(1);
                }
                skip_stats = true;
            }
        }
    } else {
    std::cerr << "\nUnacceptable number of arguments. Exiting...";
    exit(1);
    }
}


int main(int argc, char* argv[]) {
    ParseCmdArgs(argc, argv);

    std::string dataset_path = "training_set";
    auto names = GetNames(dataset_path);
    auto dataset = LoadDataset(names, dataset_path);
    names.push_back("Unknown");

    double true_positives = 0;
    double false_positives = 0;
    double false_negatives = 0;
    std::vector<std::vector<double>> classes_statistics(3, std::vector<double>(names.size(), 0));

    if (single_video_flag) {
        video_path = "test\\" + video_name;
        capture = cv::VideoCapture(video_path);
        FaceRecognition(
            video_path, names, capture, dataset, true_positives, false_positives, false_negatives,
            classes_statistics, skip_stats);
    } else {
        std::vector<std::string> directories;
        for(auto& video: std::filesystem::directory_iterator("test")) {
            if (!video.is_directory()) {
                video_path = video.path().string();
                capture = cv::VideoCapture(video_path);
                FaceRecognition(
                    video_path, names, capture, dataset, true_positives, false_positives, false_negatives,
                    classes_statistics, skip_stats);
            }      
        }
    }
    if (!skip_stats) {
        PrintDetectionStatistics(true_positives, false_positives, false_negatives);
        PrintClassesStatistics(classes_statistics, names);
    }  
    return 0;
}