#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#include "model.h"
#include "VideoProcessing.h"

#include <opencv2/opencv.hpp>

cv::VideoCapture capture;

int main() {
    std::string dataset_path = "..\\..\\..\\training_set";
    auto names = GetNames(dataset_path);

    std::cout << "Training the model...\n";
    cv::Mat k_centers;
    cv::Ptr<cv::ml::LogisticRegression> classifier = cv::ml::LogisticRegression::create();
    TrainPersonClassifier(names, dataset_path, classifier, k_centers);
    std::cout << "Training complete.\n";

    std::string filename = "..\\..\\..\\test\\bateman_interrogation_cut.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename, names, classifier, k_centers, capture);
    return 0;
}