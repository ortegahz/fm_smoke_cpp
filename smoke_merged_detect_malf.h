#ifndef SMOKE_MERGED_DETECT_MALF_H
#define SMOKE_MERGED_DETECT_MALF_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

class SmokeMergedDetectMalf {
public:
    SmokeMergedDetectMalf(int bg_history_len = 30, int contrast = 5, int vid_stride = 25, bool nb_flag = true,
                          int wait_time = 0, bool shown_mode = false, bool save_mode = false, bool malf_mode = true,
                          int phase_pos = 0, string save_dir = "smd");

    void run(vector<String> &videos);

private:
    void malf_reset();

    Mat apply_laplacian(const Mat &gray_frame);

    void malf_process(const cv::Mat &frame1, const cv::Mat &frame2, int bs = 16);

    void compare_variance_frames(const Mat &var_frame1, const Mat &var_frame2, Mat &result, int max_value);

    cv::Mat letterbox(const cv::Mat &im, int new_shape = 1280, const cv::Scalar &color = cv::Scalar(114, 114, 114),
                      bool auto_ = true, bool scaleFill = false, bool scaleup = true, int stride = 32);

    void calculate_block_variances(const Mat &lap_frame, Mat &variances, Size block_size);

    // Class members
    int contrast;
    int bg_history_len;
    int vid_stride;
    int phase_pos;
    bool nb;
    int wait_time;
    bool malf_mode;

    // Malfunction state variables
    int invis_count;
    int invis_recover_count;
    bool invis_flag;
    bool blackout;
    bool exposed_flag;
    bool dark_flag;
    int blackout_count;
};

#endif // SMOKE_MERGED_DETECT_MALF_H
