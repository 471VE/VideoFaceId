#include "model.h"

#include <filesystem>

std::vector<std::string> GetNames(const std::string& path_to_dir) {
    std::vector<std::string> directories;
    for(auto& element: std::filesystem::directory_iterator(path_to_dir))
        if (element.is_directory()) {
            std::string full_path = element.path().string();
            std::string short_path = full_path.substr(full_path.find_last_of("\\") + 1, full_path.size());
            directories.push_back(short_path);
        }
    return directories;
}

void LoadDatasetSIFT(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Mat& all_descriptors, std::vector<cv::Mat>& all_descriptors_by_image,
    std::vector<int>& all_classes_labels, int& all_images_num)
{
    all_images_num = 0;
    for (int i = 0; i < names.size(); ++i) {
        auto path_to_person_descriptors = dataset_path + "\\" + names[i] + "\\descriptors\\SIFT";

        for(auto& descriptor_path: std::filesystem::directory_iterator(path_to_person_descriptors)) {
            auto tmp = cv::imread(descriptor_path.path().string(), cv::IMREAD_GRAYSCALE);
            cv::Mat_<float> tmp_float;
            tmp.convertTo(tmp_float, CV_32F);

            all_descriptors.push_back(tmp_float);
            all_descriptors_by_image.push_back(tmp_float);
            all_classes_labels.push_back(i);
            all_images_num++;
        }
    }
}

void TrainPersonClassifier(
    const std::vector<std::string>& names, const std::string& dataset_path,
    cv::Ptr<cv::ml::LogisticRegression>& classifier, cv::Mat& k_centers)
{
    cv::Mat all_descriptors;
    std::vector<cv::Mat> all_descriptors_by_image;
    std::vector<int> all_classes_labels;
    int all_images_num;

    LoadDatasetSIFT(names, dataset_path, all_descriptors, all_descriptors_by_image, all_classes_labels, all_images_num);

    int cluster_count = DICT_SIZE;
    int attempts = 5;
    int max_iterations_num = static_cast<int>(1e4);

    cv::Mat k_labels;
    kmeans(
        all_descriptors, cluster_count, k_labels,
        cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, max_iterations_num, 1e-4),
        attempts, cv::KMEANS_PP_CENTERS, k_centers);

    cv::Mat input_data;
    cv::Mat input_data_labels;

    for (int i = 0; i < all_images_num; i++) {
        cv::BFMatcher matcher;
        std::vector<cv::DMatch> matches;
        matcher.match(all_descriptors_by_image[i], k_centers, matches);

        cv::Mat feature_vector = cv::Mat::zeros(1, DICT_SIZE, CV_32F);
        int index = 0;
        for (auto j = matches.begin(); j < matches.end(); j++, index++) {
            feature_vector.at<float>(0, matches.at(index).trainIdx) = feature_vector.at<float>(0, matches.at(index).trainIdx) + 1;
        }

        input_data.push_back(feature_vector);
        input_data_labels.push_back(cv::Mat(1, 1, CV_32F, all_classes_labels[i]));
    }

    classifier->setLearningRate(0.001);
    classifier->setIterations(100);
    classifier->setRegularization(cv::ml::LogisticRegression::REG_L2);
    classifier->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
    classifier->setMiniBatchSize(100);

    cv::Ptr<cv::ml::TrainData> training_data = cv::ml::TrainData::create(input_data, cv::ml::ROW_SAMPLE, input_data_labels);
    classifier->train(training_data);
}

cv::Mat FaceFeatureVector(const cv::Mat& descriptors, const cv::Mat& k_centers) {
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors, k_centers, matches);

    cv::Mat feature_vector = cv::Mat::zeros(1, DICT_SIZE, CV_32F);
    int index = 0;
    for (auto j = matches.begin(); j < matches.end(); j++, index++) {
        feature_vector.at<float>(0, matches.at(index).trainIdx) = feature_vector.at<float>(0, matches.at(index).trainIdx) + 1;
    }
    return feature_vector;
}

void ComputeClassesProbabilities(cv::Mat& logit_mat) {
    float sum = 0;
    for (int i = 0; i < logit_mat.cols; ++i) {
        logit_mat.at<float>(0, i) = exp(logit_mat.at<float>(0, i));
        sum += logit_mat.at<float>(0, i);
    }
    for (int i = 0; i < logit_mat.cols; ++i) {
        logit_mat.at<float>(0, i) /= sum;
    }        
}