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
#include <opencv2/tracking/tracking.hpp>

#include <chrono>
#include <time.h>
#include <Windows.h>

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

cv::VideoCapture capture;

/*void simple_track(){
    // Types: "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();

    cv::Mat frame;
    capture >> frame;
    cv::Rect bbox(200, 170, 160, 150);
    rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
    std::string info;
    int number_of_frames = 0;

    tracker->init(frame, bbox);

    while (true)
    {
        capture >> frame;
        if (frame.empty())
            break;

        if (tracker->update(frame, bbox)) {
            info = "(" + std::to_string(bbox.x) + ", " + std::to_string(bbox.y) + ", " + std::to_string(bbox.width)
                   + ", " + std::to_string(bbox.height) + ")";
            putText(frame, info, cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
            rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
            number_of_frames++;            
        } else
            putText(frame, "Tracking failure detected", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);

        putText(frame, std::to_string(number_of_frames), cv::Point(100, 110), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        std::stringstream ss;
        ss << capture.get(cv::CAP_PROP_POS_FRAMES);
        
        imshow("Tracking", frame);

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    std::cout << number_of_frames << std::endl;
    cv::waitKey();
}*/



/*void haar_cascade(std::string img_path) {
    cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");
    cv::CascadeClassifier eyes_cascade("haarcascade_eye_tree_eyeglasses.xml");

    cv::Mat example_pic = cv::imread(img_path);
    cv::resize(example_pic, example_pic, cv::Size(), 0.25, 0.25);

    cv::Mat frame_gray;
    cvtColor(example_pic, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(example_pic, center, cv::Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4);
        cv::Mat faceROI = frame_gray(faces[i]);

        std::vector<cv::Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++)
        {
            cv::Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(example_pic, eye_center, radius, cv::Scalar(255, 0, 0), 4);
        }
    }

    show("Face detection", example_pic);
}*/


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

void FrameProcessing(
    cv::Mat& full_frame,
    cv::Mat& frame_gray,
    const double& scale_inverse,
    const int& scale,
    const int& frame_count)
{
    cv::resize(full_frame, frame_gray, cv::Size(), scale_inverse, scale_inverse);
    cvtColor(frame_gray, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

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


void FaceRecognition(std::string filename) {
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

    while (true) {
        frame_count++;
        capture >> full_frame;
        FrameProcessing(full_frame, frame_gray, scale_inverse, scale, frame_count);
        imshow("Face Detection", full_frame);

        if (first_frame)
            StartAudioPlayback(filename, video_start, first_frame);

        RealTimeSyncing(
            video_start, frame_end, capture, total_time_actual, total_time_predicted,
            frame_time, full_frame, frame_count, wait_time, keyboard);

        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}


/*void face_track() {
    
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    cv::CascadeClassifier cascade("../../../haarcascade/haarcascade_frontalface_alt.xml");
    cv::Rect face_rect;
    cv::Mat ref, gray;

    bool init_tracker = false;

    while (true) {
        capture >> ref;
        cv::resize(ref, ref, cv::Size(), 0.25, 0.25);

        if (!init_tracker) {
            cvtColor(ref, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);
            std::vector<cv::Rect> faces;
            cascade.detectMultiScale(gray, faces);

            if (faces.size() == 0) {
                putText(ref, "Cannot detect face", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);

                if (!face_rect.empty()) {
                    tracker->init(ref, face_rect);
                    init_tracker = true;
                }
            }
        }
        else {
            if (tracker->update(ref, face_rect))
                rectangle(ref, face_rect, cv::Scalar(255, 0, 0), 2, 1);
            else
                putText(ref, "Tracking failure detected", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
        }

        imshow("Frame", ref);

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
}*/

int main() {
    std::string filename = "../../../test/ford_gosling.mp4";
    capture = cv::VideoCapture(filename);
    FaceRecognition(filename);
    return 0;
}