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

    if (single_video_flag) {
        video_path = "test\\" + video_name;
        capture = cv::VideoCapture(video_path);
        FaceRecognition(video_path, names, capture, dataset);
    } else {
        std::vector<std::string> directories;
        for(auto& video: std::filesystem::directory_iterator("test")) {
            video_path = video.path().string();
            capture = cv::VideoCapture(video_path);
            FaceRecognition(video_path, names, capture, dataset);
        }
    }
    
    return 0;
}