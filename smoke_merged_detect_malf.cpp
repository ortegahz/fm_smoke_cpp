#include "smoke_merged_detect_malf.h"

SmokeMergedDetectMalf::SmokeMergedDetectMalf(int bg_history_len, int contrast, int vid_stride, bool nb_flag,
                                             int wait_time, bool shown_mode, bool save_mode, bool malf_mode,
                                             int phase_pos, std::string save_dir)
        : contrast(contrast),
          bg_history_len(bg_history_len),
          vid_stride(vid_stride),
          phase_pos(phase_pos),
          nb(nb_flag),
          wait_time(wait_time),
          malf_mode(malf_mode) {
    if (malf_mode) {
        this->bg_history_len = 75;
        malf_reset();
    }
}

void SmokeMergedDetectMalf::malf_reset() {
    invis_count = 0;
    invis_recover_count = 0;
    invis_flag = false;
    blackout = false;
    exposed_flag = false;
    dark_flag = false;
    blackout_count = 0;
}

cv::Mat SmokeMergedDetectMalf::apply_laplacian(const cv::Mat &gray_frame) {
    cv::Mat lap;
    cv::Laplacian(gray_frame, lap, CV_64F);
    return lap;
}

void SmokeMergedDetectMalf::malf_process(const cv::Mat &frame1, const cv::Mat &frame2, int bs) {
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

void
SmokeMergedDetectMalf::compare_variance_frames(const cv::Mat &var_frame1, const cv::Mat &var_frame2, cv::Mat &result,
                                               int max_value) {
    cv::Mat res;
    cv::divide(var_frame1, var_frame2 + 10, res);
    res.setTo(0.9, res < 0.9);
    cv::log(res + 0.1, res);
    res *= 30;
    res.setTo(0, res < 0);
    res.setTo(max_value, res > max_value);
    result = res;
}

cv::Mat SmokeMergedDetectMalf::letterbox(const cv::Mat &im, int new_shape, const cv::Scalar &color,
                                         bool auto_, bool scaleFill, bool scaleup, int stride) {
    cv::Mat output;
    cv::Size shape = im.size();
    int width = shape.width;
    int height = shape.height;

    cv::Size new_size;
    if (new_shape <= 0) {
        new_size = cv::Size(new_shape, new_shape);
    } else {
        new_size = cv::Size(new_shape, new_shape);
    }

    float r = std::min((float) new_size.height / height, (float) new_size.width / width);
    if (!scaleup) {
        r = std::min(r, 1.0f);
    }

    int new_unpad_width = std::round(width * r);
    int new_unpad_height = std::round(height * r);

    float ratio = static_cast<float>(new_unpad_width) / width;
    cv::Size new_unpad(new_unpad_width, new_unpad_height);

    float dw = new_size.width - new_unpad.width;
    float dh = new_size.height - new_unpad.height;

    if (auto_) {
        dw = std::fmod(dw, stride);
        dh = std::fmod(dh, stride);
    } else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_unpad = new_size;
        ratio = static_cast<float>(new_unpad.width) / width;
    }

    dw /= 2;
    dh /= 2;

    if (new_unpad != shape) {
        cv::resize(im, output, new_unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        output = im;
    }

    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));

    cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return output;
}

void
SmokeMergedDetectMalf::calculate_block_variances(const cv::Mat &lap_frame, cv::Mat &variances, cv::Size block_size) {
    int variance_rows = lap_frame.rows / block_size.height;
    int variance_cols = lap_frame.cols / block_size.width;
    variances = cv::Mat::zeros(variance_rows, variance_cols, CV_64F);

    for (int i = 0; i < variance_rows; ++i) {
        for (int j = 0; j < variance_cols; ++j) {
            int start_x = j * block_size.width;
            int start_y = i * block_size.height;
            cv::Rect block_area(start_x, start_y, block_size.width, block_size.height);

            if (start_x + block_size.width <= lap_frame.cols && start_y + block_size.height <= lap_frame.rows) {
                cv::Mat block = lap_frame(block_area);
                cv::Scalar mean, stddev;
                cv::meanStdDev(block, mean, stddev);

                variances.at<double>(i, j) = stddev[0] * stddev[0];
            }
        }
    }
}

void SmokeMergedDetectMalf::run(std::vector<std::string> &videos) {
    for (const std::string &video_path: videos) {
        std::cout << "Processing: " << video_path << std::endl;
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video file." << std::endl;
            continue;
        }

        int frame_count = 0;
        cv::Mat frame, gray, bgimage, fgmask;
        cv::Ptr<cv::BackgroundSubtractor> bg_model = cv::createBackgroundSubtractorMOG2(bg_history_len, 30, false);

        while (cap.isOpened()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, vid_stride * frame_count + phase_pos);

            if (!cap.read(frame)) {
                break;
            }

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            gray = letterbox(gray);
            bg_model->apply(gray, fgmask);
            bg_model->getBackgroundImage(bgimage);

            if (frame_count > 5) {
                malf_process(gray, bgimage, 32);
            }
            frame_count++;

            if (wait_time != 0) {
                cv::waitKey(wait_time);
            }
        }
    }
}
