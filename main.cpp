#include <vector>
#include <iostream>
#include <string>
#include <filesystem>

#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>

cv::VideoCapture capture;

bool single_video_flag = false;
std::string video_name;
std::string video_path;

void ParseCmdArgs(int argc, char** argv) {
    if (argc != 1) {
        if (argc == 3) {
            if (std::string(argv[1]) != "--video") {
                std::cerr << "\nOnly the name of the video may be specified, check the command. Exiting...";
                exit(1);
            } else {
                single_video_flag = true;
                video_name = std::string(argv[2]);
            }
        } else {
            std::cerr << "\nUnacceptable number of arguments. Exiting...";
            exit(1);
        }
    }
}

int main(int argc, char* argv[]) {
    ParseCmdArgs(argc, argv);

    std::string dataset_path = "training_set";
    auto names = GetNames(dataset_path);
    auto dataset = LoadDataset(names, dataset_path);

    double true_positives = 0;
    double false_positives = 0;
    double false_negatives = 0;

    if (single_video_flag) {
        video_path = "test\\" + video_name;
        capture = cv::VideoCapture(video_path);
        FaceRecognition(video_path, names, capture, dataset, true_positives, false_positives, false_negatives);
    } else {
        std::vector<std::string> directories;
        for(auto& video: std::filesystem::directory_iterator("test")) {
            if (!video.is_directory()) {
                video_path = video.path().string();
                capture = cv::VideoCapture(video_path);
                FaceRecognition(video_path, names, capture, dataset, true_positives, false_positives, false_negatives);
            }      
        }
    }

    double precision = true_positives / (true_positives + false_positives);
    double recall = true_positives / (true_positives + false_negatives);
    double FNR = 1 - recall;

    std::cout << "Precision: " << precision << ".\n";
    std::cout << "Recall: " << recall << ".\n";
    std::cout << "FNR: " << FNR << ".\n\n";
    
    return 0;
}