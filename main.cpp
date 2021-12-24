#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <filesystem>

#include "model.h"
#include "VideoProcessing.h"
#include "matrix_read_write.h"

#include <opencv2/opencv.hpp>

cv::VideoCapture capture;

enum mode {TRAIN, IDENTIFICATION, TRAIN_AND_ID};
mode current_mode;

bool single_video_flag = false;
std::string video_name;
std::string video_path;

void ParseCmdArgs(int argc, char** argv) {
    if (argc == 3 || argc == 5) {
        if (std::string(argv[1]) != "--mode") {
            std::cerr << "\nOnly program mode and (optionally) name of the video must be specified, check the command. Exiting...";
            exit(1);
        }
        if (std::string(argv[2]) == "train") current_mode = TRAIN;
        else if (std::string(argv[2]) == "identification") current_mode = IDENTIFICATION;
        else if (std::string(argv[2]) == "train-and-identification") current_mode = TRAIN_AND_ID;
        else {
            std::cerr << "\nUnacceptable mode. Exiting...";
            exit(1);
        }
        if (argc == 5) {
            if (std::string(argv[3]) != "--video") {
                std::cerr << "\nOnly program mode and (optionally) name of the video must be specified, check the command. Exiting...";
                exit(1);
            } else if (current_mode == TRAIN)
                std::cout << "\nSince videos are not processed in \"train\" mode, \"--video\" flag will be ignored.\n\n";
            else {
                single_video_flag = true;
                video_name = std::string(argv[4]);
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

    cv::Mat k_centers;
    cv::Ptr<cv::ml::LogisticRegression> classifier;
    cv::Mat classifier_params;

    if (current_mode == TRAIN || current_mode == TRAIN_AND_ID) {
        std::cout << "Training the model...\n";
        classifier = cv::ml::LogisticRegression::create();
        TrainPersonClassifier(names, dataset_path, classifier, k_centers);
        std::cout << "Training complete.\n";

        SaveMatBinary("classifier\\k_centers.bin", k_centers);
        classifier_params = classifier->get_learnt_thetas();
        SaveMatBinary("classifier\\classifier_params.bin", classifier_params);
        std::cout << "The model was saved successfully.\n";
    }

    if (current_mode == IDENTIFICATION) {
        LoadMatBinary("classifier\\k_centers.bin", k_centers);
        LoadMatBinary("classifier\\classifier_params.bin", classifier_params);
    }

    if (current_mode == IDENTIFICATION || current_mode == TRAIN_AND_ID) {
        if (single_video_flag) {
            video_path = "test\\" + video_name;
            capture = cv::VideoCapture(video_path);
            FaceRecognition(video_path, names, classifier_params, k_centers, capture);
        } else {
            std::vector<std::string> directories;
            for(auto& video: std::filesystem::directory_iterator("test")) {
                video_path = video.path().string();
                capture = cv::VideoCapture(video_path);
                FaceRecognition(video_path, names, classifier_params, k_centers, capture);
            }
        }
    }
    
    return 0;
}