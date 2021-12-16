#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/ml/ml.hpp>

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/tracking.hpp>

#include <chrono>
#include <time.h>
#include <Windows.h>

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

cv::VideoCapture capture;

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
    int& keyboard)
{
    frame_end = Time::now();
    total_time_actual = std::chrono::duration_cast<ns>(frame_end - video_start).count() * 1e-6;
    total_time_predicted = frame_time * static_cast<double>(frame_count);
    if (total_time_actual > total_time_predicted) {
        if (total_time_actual > total_time_predicted + frame_time) {
            // Skip frames if program runs too slow
            int excess_frames = static_cast<int>((total_time_actual - total_time_predicted) / frame_time);
            for (int i = 0; i < excess_frames; ++i) {
                capture >> full_frame;
                frame_count++;
            }
        }
        // Immediately go to the next frame if delay is less than one frame or necesary frames have already been skipped
        keyboard = cv::waitKey(1);
    } else {
        // Wait if processing is ahead of audio
        wait_time = cv::saturate_cast<uchar>(total_time_predicted - total_time_actual) + 1;
        keyboard = cv::waitKey(wait_time);
    }
}


void StartAudioPlayback(const std::string& filename, Time::time_point& video_start, bool& first_frame) {
    std::string mci_string = "open " + filename + " type mpegvideo alias AudioFile";
    LPCSTR mci_command = mci_string.c_str();
    mciSendString(mci_command, 0, 0, 0);
    mciSendString("play AudioFile from 0", 0, 0, 0);
    video_start = Time::now();
    mciSendString("window AudioFile state hide", 0, 0, 0);
    first_frame = false;
}


cv::CascadeClassifier face_cascade("../../../haarcascade/haarcascade_frontalface_alt.xml");

void Preprocessing(const cv::Mat& full_frame, cv::Mat& frame_gray, const double& scale_inverse) {
    cv::resize(full_frame, frame_gray, cv::Size(), scale_inverse, scale_inverse);
    cvtColor(frame_gray, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
}


void DrawFaces(cv::Mat full_frame, const std::vector<cv::Rect>& faces, const int& scale) {
    if (faces.size() > 0) {
        for (size_t i = 0; i < faces.size(); i++) {
            cv::rectangle(
                full_frame, cv::Rect(
                    faces[i].x*scale, faces[i].y*scale,
                    faces[i].width*scale, faces[i].height*scale),
                cv::Scalar(255, 0, 0), 2, 1);
        }
    }
}

cv::Ptr<cv::Tracker> ReturnTracker(const std::string& tracker_type) {
    // These are all the trackers present in non-legacy OpenCV 4.5.4
    if (tracker_type == "MIL")
        return cv::TrackerMIL::create();
    else if (tracker_type == "KCF")
        return cv::TrackerKCF::create();
    else if (tracker_type == "CSRT")
        return cv::TrackerCSRT::create();
    else {
        std::cout << "Unknown tracker type. ";
        std::cout << "Check if the tracker type is in the list of acceptable trackers and if there are any typos. Exiting...\n";
        exit(1);
    }
}

class MultiTracker {
    public:
        MultiTracker(const std::string& tracker_type) {
            tracker_type_ = tracker_type;
        };

        void start(const cv::Mat& frame, const std::vector<cv::Rect>& faces) {
            num_trackers_ = faces.size();
            trackers_list_.clear();
            trackers_list_.insert(
                trackers_list_.end(),
                num_trackers_,
                ReturnTracker(tracker_type_));
            for (size_t i = 0; i < num_trackers_; ++i) {
                trackers_list_[i]->init(frame, faces[i]);
            }
        }

        void update(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
            for (size_t i = 0; i < num_trackers_; ++i) {
                trackers_list_[i]->update(frame, faces[i]);
            }
        }

    private:
        std::vector<cv::Ptr<cv::Tracker>> trackers_list_;
        std::string tracker_type_;
        size_t num_trackers_;
};


void FaceRecognition(std::string filename, const std::string& tracker_type = "NO_TRACKER") {
    double frame_time = 1000. / capture.get(cv::CAP_PROP_FPS);
    Time::time_point video_start, frame_end;
    uchar wait_time;
    int keyboard = 0;

    cv::Mat full_frame, frame_gray;
    int scale = 4;
    double scale_inverse = 1. / static_cast<double>(scale);

    bool first_frame = true;
    int frame_count = 0;
    double total_time_actual, total_time_predicted;
    std::vector<cv::Rect> faces;

    MultiTracker trackers(tracker_type);

    while (true) {
        capture >> full_frame;
        if (full_frame.empty())
            break;
        frame_count++;
        
        Preprocessing(full_frame, frame_gray, scale_inverse);

        if (tracker_type != "NO_TRACKER") {
            if (frame_count % 10 == 1) {
                face_cascade.detectMultiScale(frame_gray, faces);
                trackers.start(frame_gray, faces);
            }
            else {
                trackers.update(frame_gray, faces);
            }
        }
        else
            face_cascade.detectMultiScale(frame_gray, faces);

        DrawFaces(full_frame, faces, scale);
        cv::imshow("Face Detection", full_frame);

        if (first_frame)
            StartAudioPlayback(filename, video_start, first_frame);

        RealTimeSyncing(
            video_start, frame_end, capture, total_time_actual, total_time_predicted,
            frame_time, full_frame, frame_count, wait_time, keyboard);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}


int main() {
    std::string filename = "../../../test/ford_gosling.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename);
    return 0;
}