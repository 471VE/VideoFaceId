#ifndef MATRIX_READ_WRITE_H
#define MATRIX_READ_WRITE_H

#include <opencv2/opencv.hpp>
#include <fstream>

void check_directory(std::string directoryName);

bool WriteMatBinary(std::ofstream& ofs, const cv::Mat& out_mat);
bool SaveMatBinary(const std::string& filename, const cv::Mat& output);

bool ReadMatBinary(std::ifstream& ifs, cv::Mat& in_mat);
bool LoadMatBinary(const std::string& filename, cv::Mat& output);

#endif // MATRIX_READ_WRITE_H