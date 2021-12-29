#include "VideoProcessing.h"
#include "MultiTracker.h"

#include <fstream>
#include <sstream>
#include <filesystem>

#ifdef _WIN32
    #include <Windows.h>
#endif

const double GOOD_MATCH_RATIO = 0.65; //  FINETUNED
const size_t SMALL_MATCHES_NUMBER = 9; // FINETUNED

const double IoU_THRESHOLD = 0.3;


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
    #ifdef _WIN32
        std::string open_string = "open " + filename + " type mpegvideo";
        LPCSTR open_command = open_string.c_str();

        std::string play_string = "play " + filename + " from 0";
        LPCSTR play_command = play_string.c_str();

        std::string hide_string = "window " + filename + " state hide";
        LPCSTR hide_command = hide_string.c_str();

        mciSendString(open_command, 0, 0, 0);
        mciSendString(play_command, 0, 0, 0);
        mciSendString(hide_command, 0, 0, 0);
    #endif

    video_start = Time::now();
    first_frame = false;
}

#ifdef _WIN32
void StopAudioPlayback(const std::string& filename) {
    std::string close_string = "close " + filename;
    LPCSTR close_command = close_string.c_str();
    mciSendString(close_command, 0, 0, 0);
}
#endif


void DrawFaces(
    cv::Mat full_frame,
    const std::vector<cv::Rect>& faces,
    const std::vector<size_t> name_indices_of_detected_faces,
    const std::vector<std::string>& names)
{
    for (int i = 0; i < faces.size(); ++i) {
        cv::rectangle(
            full_frame, faces[i],
            cv::Scalar(0, 0, 255), 2, 1);
        cv::putText(
            full_frame, names[name_indices_of_detected_faces[i]], cv::Point2d(faces[i].x, (faces[i].y + faces[i].height) + 25),
            cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
}


void match(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& good_matches) {
    good_matches.clear();

    cv::BFMatcher desc_matcher(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> vmatches;
    desc_matcher.knnMatch(desc1, desc2, vmatches, 2);

    for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
        if (!vmatches[i].size()) {
            continue;
        }
        if (vmatches[i][0].distance < GOOD_MATCH_RATIO * vmatches[i][1].distance)
            good_matches.push_back(vmatches[i][0]);
    }
}


void FaceIdentification(
    std::vector<size_t>& name_indices_of_detected_faces,
    const std::vector<std::string>& names,
    const std::vector<cv::Rect>& faces,
    cv::Ptr<cv::SIFT>& detector,
    const cv::Mat& full_frame,
    std::vector<cv::KeyPoint>& person_keypoints_tmp,
    cv::Mat& person_descriptors,    
    std::vector<cv::DMatch>& good_matches,
    std::vector<std::pair<size_t, size_t>>& good_matches_num,
    const std::vector<std::vector<cv::Mat>>& dataset)
{
    name_indices_of_detected_faces.clear();
    for (const auto& face: faces) {
        detector->detectAndCompute(full_frame(face), cv::Mat(), person_keypoints_tmp, person_descriptors);    
        for (size_t i = 0; i < dataset.size(); ++i) {
            for (const auto& true_descriptors: dataset[i]) {
                match(person_descriptors, true_descriptors, good_matches);
                good_matches_num.push_back(std::make_pair(good_matches.size(), i));
            }
        }
        std::sort(good_matches_num.begin(), good_matches_num.end(), std::greater<>());

        if (good_matches_num[0].first < SMALL_MATCHES_NUMBER)
            name_indices_of_detected_faces.push_back(names.size() - 1);
        else
            name_indices_of_detected_faces.push_back(good_matches_num[0].second);
    }
}


cv::CascadeClassifier face_cascade("haarcascade/haarcascade_frontalface_alt2.xml");

void TrackOrDetect(
    const std::string& tracker_type,
    const int& frame_count,
    std::vector<cv::Rect>& faces,
    const cv::Mat& frame_downscaled,
    const cv::Mat& frame_gray,
    MultiTracker& trackers,
    bool& tracked)
{
    if (tracker_type != "NO_TRACKER") {
        if (frame_count % 20 == 1 || faces.size() == 0) {
            faces.clear();
            face_cascade.detectMultiScale(frame_gray, faces);
            trackers.start(frame_downscaled, faces);
        }
        else {
            if (!trackers.update(frame_downscaled, faces)) {
                faces.clear();
                face_cascade.detectMultiScale(frame_gray, faces);
                trackers.start(frame_downscaled, faces);
            } else
                tracked = true;
        }
    }
    else
        face_cascade.detectMultiScale(frame_downscaled, faces);
}


std::vector<std::vector<cv::Mat>> LoadDataset(const std::vector<std::string>& names, const std::string& dataset_path) {
    std::vector<cv::Mat> person_descriptors;
    std::vector<std::vector<cv::Mat>> dataset;
    cv::Mat_<float> tmp, tmp_float;
    for (const auto& name: names) {
        person_descriptors.clear();
        auto path_to_person_descriptors = dataset_path + "\\" + name + "\\descriptors\\SIFT";
        for(auto& descriptor_path: std::filesystem::directory_iterator(path_to_person_descriptors)) {
            tmp = cv::imread(descriptor_path.path().string(), cv::IMREAD_GRAYSCALE);
            tmp.convertTo(tmp_float, CV_32F);
            person_descriptors.push_back(tmp_float);
        }
        dataset.push_back(person_descriptors);
    }
    return dataset;
}


double InetersectionOverUnion(const cv::Rect& rectangleA, const cv::Rect& rectangleB) {
    double rect_intersection = (rectangleA & rectangleB).area();
    double rect_union = rectangleA.area() + rectangleB.area() - rect_intersection;
    return rect_intersection / rect_union;
}


bool AreTheSameFace(const cv::Rect& rectangleA, const cv::Rect& rectangleB) {
    double IoU = InetersectionOverUnion(rectangleA, rectangleB);
    if (IoU > IoU_THRESHOLD) {
        return true;
    }
    return false;
}


void CheckFacesForIntersection(std::vector<cv::Rect>& faces) {
    std::vector<int> to_delete;
    for (int i = 0; i < faces.size(); ++i)
        for (int j = i+1; j < faces.size(); ++j)
            if (AreTheSameFace(faces[i], faces[j]))
                to_delete.push_back(j);
    for (int i = static_cast<int>(to_delete.size()) - 1; i >= 0; --i) 
        faces.erase(faces.begin() + to_delete[i]);
}


void LoadAnnotationsSingleFile(
    const std::string& videoname,
    std::vector<std::vector<cv::Rect>>& annotation_rectangles,
    std::vector<std::vector<bool>>& annotation_mask,
    std::vector<int>& name_indices)
{
    std::string short_name = videoname.substr(
        videoname.find_last_of("\\") + 1,
        videoname.size() - videoname.find_last_of("\\") - 5);
    std::string annotation_folder = "test\\annotations\\" + short_name;
    annotation_rectangles.clear();
    annotation_mask.clear();
    name_indices.clear();

    std::vector<std::string> annotation_filenames;
    std::vector<cv::Rect> rectangle_vector;

    std::string file_line;
    std::string parameter;

    cv::Rect rectangle;
    bool rectangle_exists;
    std::vector<bool> rectangles_exist;
    

    for (auto& element: std::filesystem::directory_iterator(annotation_folder))
        annotation_filenames.push_back(element.path().string());
    
        for (size_t i = 0; i < annotation_filenames.size(); ++i) {
            rectangles_exist.clear();
            rectangle_vector.clear();

            std::ifstream annotation_file(annotation_filenames[i]);
            std::getline(annotation_file, file_line);
            name_indices.push_back(std::stoi(file_line));

            while (std::getline(annotation_file, file_line)) {
                std::istringstream input;
                input.str(file_line);
                std::vector<int> square_params;
                while (std::getline(input, parameter, ' ')) {
                    square_params.push_back(std::stoi(parameter));
                }
                if (square_params.size() == 3) {
                    rectangle = cv::Rect(square_params[0], square_params[1], square_params[2], square_params[2]);
                    rectangle_exists = true;
                } else {
                    rectangle = cv::Rect(0, 0, 0, 0);
                    rectangle_exists = false;
                }
                rectangles_exist.push_back(rectangle_exists);
                rectangle_vector.push_back(rectangle);
            }
        annotation_file.close();
        annotation_rectangles.push_back(rectangle_vector);
        annotation_mask.push_back(rectangles_exist);
    }
} 


void FrameDetectedStatistics(
    const std::vector<cv::Rect>& true_faces,
    const std::vector<cv::Rect>& detected_faces,
    double& true_positives,
    double& false_positives,
    double& false_negatives)
{
    double true_positive = 0;
    double false_positive = static_cast<double>(detected_faces.size());
    double false_negative = static_cast<double>(true_faces.size());
    bool skip = false;

    std::vector<cv::Rect> true_faces_copy = true_faces;

    for (int i = 0; i < detected_faces.size(); ++i) {
        for (int j = 0; j < true_faces_copy.size(); ++j) {
            if (AreTheSameFace(true_faces_copy[j], detected_faces[i])) {
                true_positive++;
                false_negative--;
                false_positive--;
                true_faces_copy.erase(true_faces_copy.begin() + j);
                skip = true;
                break;
            }
        }
        if (skip) {
            skip = false;
            continue;
        }
    }

    true_positives += true_positive;
    false_positives += false_positive;
    false_negatives += false_negative;
}


void FrameClassStatistics(
    const std::vector<size_t>& name_indices_of_true_faces,
    const std::vector<size_t>& name_indices_of_detected_faces,
    const std::vector<cv::Rect>& true_faces,
    const std::vector<cv::Rect>& detected_faces,
    std::vector<std::vector<double>>& classes_statistics)
{   
    bool skip = false;
    std::vector<cv::Rect> detected_faces_copy = detected_faces;
    std::vector<size_t> name_indices_of_detected_faces_copy = name_indices_of_detected_faces;

    for (int i = 0; i < true_faces.size(); ++i) {
        for (int j = 0; j < detected_faces_copy.size(); ++j) {        
            if (AreTheSameFace(true_faces[i], detected_faces_copy[j])) {
                if (name_indices_of_true_faces[i] == name_indices_of_detected_faces_copy[j]) {
                    classes_statistics[0][name_indices_of_true_faces[i]]++;
                    detected_faces_copy.erase(detected_faces_copy.begin() + j);
                    name_indices_of_detected_faces_copy.erase(name_indices_of_detected_faces_copy.begin() + j);
                    skip = true;
                    break;
                }
            }
        }
        if (skip) {
            skip = false;
            continue;
        } else
            classes_statistics[2][name_indices_of_true_faces[i]]++;
    }

    for (int i = 0; i < detected_faces_copy.size(); ++i)
        classes_statistics[1][name_indices_of_detected_faces_copy[i]]++;
}


void FaceRecognition(
    const std::string& filename,
    const std::vector<std::string>& names,
    cv::VideoCapture& capture,
    const std::vector<std::vector<cv::Mat>>& dataset,
    double& true_positives,
    double& false_positives,
    double& false_negatives,
    std::vector<std::vector<double>>& classes_statistics,
    const std::string& tracker_type)
{
    double frame_time = 1000. / capture.get(cv::CAP_PROP_FPS);
    Time::time_point video_start, frame_end;
    uchar wait_time;
    int keyboard = 0;

    cv::Mat full_frame, frame_downscaled, frame_gray;

    double width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    double new_width = 320;
    double new_height = 180;

    double scale = (((width/new_width) < (height/new_height)) ? (width/new_width) : (height/new_height));
    double scale_inverse = 1. / static_cast<double>(scale);

    bool first_frame = true;
    int frame_count = 0;
    double total_time_actual, total_time_predicted;

    std::vector<cv::Rect> true_faces;
    std::vector<cv::Rect> detected_faces;
    std::vector<cv::Rect> faces_downscaled;
    MultiTracker trackers = MultiTracker(tracker_type);

    auto detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> person_keypoints_tmp;
    cv::Mat person_descriptors;

    std::vector<size_t> name_indices_of_detected_faces;
    std::vector<size_t> name_indices_of_true_faces;

    std::vector<cv::DMatch> good_matches;
    std::vector<std::pair<size_t, size_t>> good_matches_num;

    std::vector<std::vector<cv::Rect>> annotation_rectangles;
    std::vector<std::vector<bool>> annotation_mask;
    std::vector<int> name_indices;
    LoadAnnotationsSingleFile(filename, annotation_rectangles, annotation_mask, name_indices);

    bool tracked;
    
    while (true) {
        capture >> full_frame;
        if (full_frame.empty())
            break;
        frame_count++;

        cv::resize(full_frame, frame_downscaled, cv::Size(), scale_inverse, scale_inverse);
        cvtColor(frame_downscaled, frame_gray, cv::COLOR_BGR2GRAY);

        tracked = false;
        TrackOrDetect(tracker_type, frame_count, faces_downscaled, frame_downscaled, frame_gray, trackers, tracked);

        if (faces_downscaled.size() > 1)
            CheckFacesForIntersection(faces_downscaled);

        detected_faces.clear();
        for (size_t i = 0; i < faces_downscaled.size(); ++i)
            detected_faces.push_back(cv::Rect2d(
                faces_downscaled[i].x*scale, faces_downscaled[i].y*scale,
                faces_downscaled[i].width*scale, faces_downscaled[i].height*scale));

        if (!tracked) {
            FaceIdentification(
                name_indices_of_detected_faces, names, detected_faces, detector, full_frame,
                person_keypoints_tmp, person_descriptors, good_matches, good_matches_num, dataset);
        }

        true_faces.clear();
        name_indices_of_true_faces.clear();
        for (size_t i = 0; i < name_indices.size(); ++i){
            if (annotation_mask[i][frame_count - 1]) {
                true_faces.push_back(annotation_rectangles[i][frame_count - 1]);
                name_indices_of_true_faces.push_back(name_indices[i]);
            }
        }
                
        FrameDetectedStatistics(true_faces, detected_faces, true_positives, false_positives, false_negatives);
        FrameClassStatistics(name_indices_of_true_faces, name_indices_of_detected_faces, true_faces, detected_faces, classes_statistics);
        DrawFaces(full_frame, detected_faces, name_indices_of_detected_faces, names);

        cv::imshow("Face Recognition and Identification", full_frame);

        if (first_frame) {
            first_frame = false;
            StartAudioPlayback(filename, video_start, first_frame);
        }
   
        RealTimeSyncing(
            video_start, frame_end, capture, total_time_actual, total_time_predicted,
            frame_time, full_frame, frame_count, wait_time, keyboard);
        
        if (keyboard == 'q' || keyboard == 27) {
            #ifdef _WIN32
                StopAudioPlayback(filename);
            #endif
            break;
        }
    }
}


void PrecisionRecallFNR(
    double& precision, double& recall, double& FNR,
    const double& true_positives, const double& false_positives, const double& false_negatives)
{
    precision = (true_positives + false_positives == 0) ? 0 : true_positives / (true_positives + false_positives);
    recall = true_positives / (true_positives + false_negatives);
    FNR = 1 - recall;
}


void PrintDetectionStatistics(const double& true_positives, const double& false_positives, const double& false_negatives) {
    double precision, recall, FNR;
    PrecisionRecallFNR(precision, recall, FNR, true_positives, false_positives, false_negatives);
    std::cout << "\n#####   DETECTION AND TRACKING STATISTICS   #####\n\n";
    std::cout << "Precision - " << precision << ".\n";
    std::cout << "Recall - " << recall << ".\n";
    std::cout << "FNR - " << FNR << ".\n";
}


void PrintClassesStatistics(const std::vector<std::vector<double>>& classes_statistics, const std::vector<std::string>& names) {
    double precision, recall, FNR;
    double true_positives, false_positives, false_negatives;
    std::cout << "\n\n##########    STATISTICS ON CLASSES    ##########\n";
    for (size_t i = 0; i < names.size(); ++i) {
        true_positives = classes_statistics[0][i];
        false_positives = classes_statistics[1][i];
        false_negatives = classes_statistics[2][i];
        if (true_positives == 0 && false_positives == 0 && false_negatives == 0)
            continue;
        PrecisionRecallFNR(precision, recall, FNR, true_positives, false_positives, false_negatives);
        std::cout << "\n" << names[i] << ":\n";
        std::cout << "Precision - " << precision << ".\n";
        std::cout << "Recall - " << recall << ".\n";
        std::cout << "FNR - " << FNR << ".\n";
    }
}