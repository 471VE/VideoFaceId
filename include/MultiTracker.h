#ifndef MULTITRACKER_H
#define MULTITRACKER_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/tracking.hpp>

// The MultiTracker class is used to track multiple objects using the specified tracker algorithm.
// It is naive implementation of multiple object tracking.
class MultiTracker {
    public:
        MultiTracker(const std::string& tracker_type);

        void start(const cv::Mat& frame, const std::vector<cv::Rect>& faces);
        bool update(const cv::Mat& frame, std::vector<cv::Rect>& faces);

    private:
        std::vector<cv::Ptr<cv::Tracker>> trackers_list_;
        std::string tracker_type_;
        size_t num_trackers_;

        cv::Ptr<cv::Tracker> ReturnTracker();
};

#endif // MULTITRACKER_H