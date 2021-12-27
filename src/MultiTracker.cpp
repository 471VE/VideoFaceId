#include "MultiTracker.h"

#include <iostream>

#include <opencv2/video.hpp>

MultiTracker::MultiTracker(const std::string& tracker_type)
    : tracker_type_(tracker_type)
{}

void MultiTracker::start(const cv::Mat& frame, const std::vector<cv::Rect>& faces) {
    num_trackers_ = faces.size();
    trackers_list_.clear();
    trackers_list_.insert(
        trackers_list_.end(),
        num_trackers_,
        ReturnTracker());
    for (size_t i = 0; i < num_trackers_; ++i) {
        trackers_list_[i]->init(frame, faces[i]);
    }
}

bool MultiTracker::update(const cv::Mat& frame, std::vector<cv::Rect>& faces) {
    for (size_t i = 0; i < num_trackers_; ++i) {
        if (!trackers_list_[i]->update(frame, faces[i])) return false;
    }
    return true;
}

cv::Ptr<cv::Tracker> MultiTracker::ReturnTracker() {
    // These are all the trackers present in OpenCV 4.5.4
    if (tracker_type_ == "MIL")
        return cv::TrackerMIL::create();

    else if (tracker_type_ == "KCF")
    {
        cv::TrackerKCF::Params params;
        params.desc_pca = cv::TrackerKCF::MODE::CN;
        params.desc_npca = cv::TrackerKCF::MODE::GRAY;
        return cv::TrackerKCF::create(params);
    }

    else if (tracker_type_ == "CSRT")
        return cv::TrackerCSRT::create();
        
    else {
        std::cout << "Unknown tracker type. ";
        std::cout << "Check if the tracker type is in the list of acceptable trackers and if there are any typos. Exiting...\n";
        exit(1);
    }
}
