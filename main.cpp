#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class SmokeMergedDetectMalf {
public:
    SmokeMergedDetectMalf(
            int bg_history_len = 30,
            int contrast = 5,
            int vid_stride = 25,
            bool nb_flag = true,
            int wait_time = 0,
            bool shown_mode = false,
            bool save_mode = false,
            bool malf_mode = true,
            int phase_pos = 0,
            string save_dir = "smd"
    ) : contrast(contrast),
        bg_history_len(bg_history_len),
        nb(nb_flag),
        wait_time(wait_time),
        malf_mode(malf_mode) {
        if (malf_mode) {
            this->bg_history_len = 75;
            malf_reset();
        }
    }

    void malf_reset() {
        invis_count = 0;
        invis_recover_count = 0;
        invis_flag = false;
        blackout = false;
        exposed_flag = false;
        dark_flag = false;
        blackout_count = 0;
    }

    Mat apply_laplacian(const Mat &gray_frame) {
        Mat lap;
        Laplacian(gray_frame, lap, CV_64F);
        return lap;
    }

    void malf_process(const cv::Mat &frame1, const cv::Mat &frame2, int bs = 16) {
        cv::Size block_size(bs, bs);
        cv::Mat lap1 = apply_laplacian(frame1);
        cv::Mat lap2 = apply_laplacian(frame2);

        cv::Mat var_frame1, var_frame2;
        calculate_block_variances(lap1, var_frame1, block_size);
        calculate_block_variances(lap2, var_frame2, block_size);

        if (malf_mode) {
            cv::Mat comparison_result;
            compare_variance_frames(var_frame2, var_frame1, comparison_result, 120);

            int comp_len = cv::countNonZero(var_frame2 >= 100);
            double bg_gray = cv::mean(frame2)[0];
            double cur_gray = cv::mean(frame1)[0];
            double comp_value = -1;

            // Correctly compute the Laplacian and its variance
            cv::Mat lap;
            cv::Laplacian(frame1, lap, CV_64F);
            double lap_overall = cv::mean(lap.mul(lap))[0];

            if (std::max(bg_gray, cur_gray) / std::max(20.0, std::min(bg_gray, cur_gray)) >= 2 || cur_gray < 30 ||
                cur_gray > 220) {
                blackout = true;
                blackout_count += 2;
                if (blackout_count >= 20) {
                    if (cur_gray < 30 && lap_overall < 500) {
                        if (!dark_flag) {
                            dark_flag = true;
                            std::cout << "Malfunction - Too dark" << std::endl;
                        }
                    } else if (cur_gray > 220 && lap_overall < 500) {
                        if (!exposed_flag) {
                            exposed_flag = true;
                            std::cout << "Malfunction - Exposed" << std::endl;
                        }
                    }
                }
                blackout_count = std::min(20, blackout_count);
            } else {
                blackout = false;
            }

            if (blackout || blackout_count != 0) {
                blackout_count--;
            } else {
                if (blackout_count == 0) {
                    if (exposed_flag) {
                        exposed_flag = false;
                        std::cout << "Remove the Malfunction - Exposed" << std::endl;
                    }
                    if (dark_flag) {
                        dark_flag = false;
                        std::cout << "Remove the Malfunction - Too dark" << std::endl;
                    }
                }
                if (comp_len > 80) {
                    cv::Scalar mean_val, stddev_val;
                    cv::meanStdDev(comparison_result, mean_val, stddev_val, var_frame2 >= 100);
                    comp_value = mean_val[0];
                    if (!invis_flag) {
                        if (comp_value < 40) {
                            invis_count = std::max(0, invis_count - 1);
                        } else {
                            invis_count++;
                            if (invis_count == 20) {
                                std::cout << "Malfunction - Loss of visibility" << std::endl;
                                invis_flag = true;
                            }
                        }
                    }
                } else {
                    if (!invis_flag) {
                        if (invis_count > 10) {
                            invis_count++;
                            if (invis_count == 20) {
                                std::cout << "Malfunction - Loss of visibility" << std::endl;
                                invis_flag = true;
                            }
                        } else {
                            invis_count--;
                        }
                    }
                }
            }

            if (invis_flag) {
                if (comp_value < 0.3 && comp_value != -1 && lap_overall > 200) {
                    invis_recover_count++;
                    if (invis_recover_count >= 10) {
                        invis_flag = false;
                        std::cout << "Remove the Malfunction - Loss of visibility" << std::endl;
                        invis_count = 0;
                        invis_recover_count = 0;
                    }
                } else {
                    invis_recover_count = 0;
                }
            }
            std::cout << "INVIS:" << invis_flag << ", EXP:" << exposed_flag << ", DAR:" << dark_flag
                      << ", B:" << blackout << ", value=" << comp_value << ", len=" << comp_len
                      << ", ic=" << invis_count << ", irc=" << invis_recover_count << ", bc=" << blackout_count
                      << std::endl;
        }
    }


    void compare_variance_frames(const Mat &var_frame1, const Mat &var_frame2, Mat &result, int max_value) {
        Mat res;
        divide(var_frame1, var_frame2 + 10, res);
        res.setTo(0.9, res < 0.9);
        log(res + 0.1, res);
        res *= 30;
        res.setTo(0, res < 0);
        res.setTo(max_value, res > max_value);
        result = res;
    }


    void run(vector<String> &videos) {
        for (const string &video_path: videos) {
            cout << "Processing: " << video_path << endl;
            VideoCapture cap(video_path);
            if (!cap.isOpened()) {
                cerr << "Error opening video file." << endl;
                continue;
            }

            int frame_count = 0;
            Mat frame, gray, bgimage;

            Ptr <BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2(bg_history_len, 30, false);

            while (cap.read(frame)) {
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                bg_model->apply(gray, bgimage);

                if (frame_count > 5) {
                    malf_process(gray, bgimage, 32);
                }
                frame_count++;
                if (wait_time != 0) {
                    waitKey(wait_time);
                }
            }
        }
    }

private:
    int contrast;
    int bg_history_len;
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

    void calculate_block_variances(const Mat &lap_frame, Mat &variances, Size block_size) {
        // 计算块的行列数量
        int variance_rows = lap_frame.rows / block_size.height;
        int variance_cols = lap_frame.cols / block_size.width;

        // 初始化variances矩阵
        // 数据类型为CV_64F，因为方差通常是双精度浮点型
        variances = Mat::zeros(variance_rows, variance_cols, CV_64F);

        for (int i = 0; i < variance_rows; ++i) {
            for (int j = 0; j < variance_cols; ++j) {
                int start_x = j * block_size.width;
                int start_y = i * block_size.height;
                Rect block_area(start_x, start_y, block_size.width, block_size.height);

                // 确保块不超过图片边界
                if (start_x + block_size.width <= lap_frame.cols && start_y + block_size.height <= lap_frame.rows) {
                    Mat block = lap_frame(block_area);
                    Scalar mean, stddev;
                    meanStdDev(block, mean, stddev);

                    // 将方差存储在variances矩阵中
                    variances.at<double>(i, j) = stddev[0] * stddev[0];
                }
            }
        }
    }

};

int main() {
    vector<string> video_paths = {"/media/manu/ST8000DM004-2U91/smoke/fault/vas2.mp4"};
    SmokeMergedDetectMalf md(75, 5, 0, false, true, false, true);
    md.run(video_paths);

    return 0;
}
