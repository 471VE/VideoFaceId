#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

const int DICT_SIZE = 1000;

std::vector<std::string> GetNames(const std::string& path_to_dir);

void LoadDatasetSIFT(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Mat& all_descriptors, std::vector<cv::Mat>& all_descriptors_by_image,
    std::vector<int>& all_classes_labels, int& all_images_num);

void TrainPersonClassifier(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Ptr<cv::ml::LogisticRegression>& classifier, cv::Mat& k_centers);

cv::Mat FaceFeatureVector(const cv::Mat& descriptors, const cv::Mat& k_centers);

void ComputeClassesProbabilities(cv::Mat& logit_mat);

#endif // MODEL_H