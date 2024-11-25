#include "smoke_merged_detect_malf.h"
#include <numeric>

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

double mean(const std::vector<double> &data) {
    if (data.empty())
        return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
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
    // cv::Mat lap2 = apply_laplacian(frame2);

    cv::Mat var_frame1, var_frame2;
    calculate_block_variances(lap1, var_frame1, block_size);
    // calculate_block_variances(lap2, var_frame2, block_size);

    if (malf_mode) {
        int blackout_countmax = 5; // 20
        int invis_countmax = 3;     // 20
        int invis_recover_countmax = 10;
        int lap_thresh = 100, gray_min_thresh = 80, gray_max_thresh = 220;
        int exposed_countmax = 5;

        std::vector<double> var_frame1_data;
        var_frame1_data.reserve(var_frame1.total());

        // var_frame1.forEach<double>([&](double &value, const int *position) -> void
        //                            { var_frame1_data.push_back(value); });

        // std::cout << "var_frame1.size() --> " << var_frame1.size() << std::endl;

        // var_frame1.forEach<double>([&](double &value, const int *position) -> void {
        //     std::cout << "Extracted value: " << value << std::endl;
        //     var_frame1_data.push_back(value);
        // });

        for (int i = 0; i < var_frame1.rows; ++i) {
            for (int j = 0; j < var_frame1.cols; ++j) {
                double value = var_frame1.at<double>(i, j);
                var_frame1_data.push_back(value);
            }
        }

        std::sort(var_frame1_data.begin(), var_frame1_data.end());

        // std::cout << "var_frame1_data.size() --> " << var_frame1_data.size() << std::endl;

        // std::cout << "All values in var_frame1_data:" << std::endl;
        // for (double value : var_frame1_data) {
        //     std::cout << value << std::endl;
        // }

        std::size_t slice_start = static_cast<std::size_t>(var_frame1_data.size() * 0.8);
        std::vector<double> top_20_percent(var_frame1_data.begin() + slice_start, var_frame1_data.end());
        double lap_value = mean(top_20_percent);

        // std::cout << "Values in top_20_percent:" << std::endl;
        // std::cout << "top_20_percent.size() --> " << top_20_percent.size() << std::endl;
        // for (double value : top_20_percent) {
        //     std::cout << value << std::endl;
        // }

        // double lap_value = 0.0;

        // LOGD("lap_value --> %f \r\n", lap_value);

        // cv::Mat comparison_result;
        // compare_variance_frames(var_frame2, var_frame1, comparison_result, 120);

        // int comp_len = cv::countNonZero(var_frame2 >= 100);
        double bg_gray = cv::mean(frame2)[0];
        double cur_gray = cv::mean(frame1)[0];
        double comp_value = -1;

        cv::Mat lap;
        cv::Laplacian(frame1, lap, CV_64F);
        double lap_overall = cv::mean(lap.mul(lap))[0];

        std::cout << ", bg_gray:" << bg_gray << ", cur_gray:" << cur_gray
                  << ", lap_value:" << lap_value << std::endl;


        // new exposed logics
        cv::Mat frame1_resized;
        cv::resize(frame1, frame1_resized, cv::Size(32, 24));

        // Convert to single channel and ensure it's 8-bit
        frame1_resized.convertTo(frame1_resized, CV_8UC1);

        // Flatten the frame to a vector
        std::vector<uchar> frame1_flatten;
        frame1_flatten.assign(frame1_resized.datastart, frame1_resized.dataend);

        // Sort the flattened vector
        std::sort(frame1_flatten.begin(), frame1_flatten.end());

        // Calculate the mean of the top 10% brightest values
        int threshold_index = static_cast<int>(frame1_flatten.size() * 0.9);
        double brighter_value = std::accumulate(frame1_flatten.begin() + threshold_index, frame1_flatten.end(), 0.0) /
                                (frame1_flatten.end() - (frame1_flatten.begin() + threshold_index));

        std::cout << "brighter_value=" << brighter_value << std::endl;

        if (!exposed_flag) {
            if (brighter_value > 230) {
                exposed_count += 1;
                if (exposed_count >= exposed_countmax) {
                    exposed_flag = true;
                    exposed_count = exposed_countmax;
                    std::cout << "Malfunction - Exposed" << std::endl;
                }
            } else {
                exposed_count = std::max(0, exposed_count - 1);
            }
        } else {
            if (brighter_value <= 200) {
                exposed_count -= 1;
                if (exposed_count <= 0) {
                    exposed_flag = false;
                    exposed_count = 0;
                    std::cout << "Remove the Malfunction - Exposed" << std::endl;
                }
            }
        }

        if (!dark_flag) {
            if (cur_gray < gray_min_thresh && lap_value < lap_thresh) {
                blackout_count += 1;
                if (blackout_count >= blackout_countmax) {
                    dark_flag = true;
                    blackout_count = blackout_countmax;
                    std::cout << "Malfunction - Dark" << std::endl;
                }
            } else {
                blackout_count = std::max(0, blackout_count - 1);
            }
        } else {
            if (cur_gray >= 100) {
                blackout_count -= 1;
                if (blackout_count <= 0) {
                    dark_flag = false;
                    blackout_count = 0;
                    std::cout << "Remove the Malfunction - Dark" << std::endl;
                }
            }
        }

        if (!invis_flag) {
            // LOGD("invis_count --> %d \r\n", invis_count);
            if (lap_value > lap_thresh) {
                invis_count = std::max(0, invis_count - 1);
            } else {
                invis_count++;
                if (invis_count == invis_countmax) {
                    std::cout << "Malfunction - Loss of visibility" << std::endl;

                    cv::Mat frame1_resized;
                    cv::resize(frame1, frame1_resized, cv::Size(32, 24));

                    frame1_resized.convertTo(frame1_resized, CV_8U);

                    double minVal, maxVal;
                    cv::minMaxLoc(frame1_resized, &minVal, &maxVal);
                    int contrast_value = static_cast<int>(maxVal - minVal);

                    std::cout << "contrast_value=" << contrast_value << std::endl;

                    if (contrast_value < 190) {
                        std::cout << "对比度损失故障报出" << std::endl;
                        contr_flag = true;
                    } else {
                        std::cout << "清晰度损失故障报出" << std::endl;
                        contr_flag = false;
                    }
                    invis_flag = true;
                }
            }
        }

        if (invis_flag) {
            if (lap_value >= 1500) {
                invis_recover_count++;
                if (invis_recover_count >= invis_recover_countmax) {
                    invis_flag = false;
                    contr_flag = false;
                    std::cout << "Remove the Malfunction - Loss of visibility" << std::endl;
                    invis_count = 0;
                    invis_recover_count = 0;
                }
            } else {
                invis_recover_count = 0;
            }
        }
        std::cout << "CONTR:" << contr_flag << "INVIS:" << invis_flag << ", EXP:" << exposed_flag << ", DAR:"
                  << dark_flag
                  << ", B:" << blackout << ", value=" << comp_value
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
        cv::Mat frame, frame_r, gray, gray_o, bgimage, fgmask;
        cv::Ptr<cv::BackgroundSubtractor> bg_model = cv::createBackgroundSubtractorMOG2(bg_history_len, 30, false);

        while (cap.isOpened()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, vid_stride * frame_count + phase_pos);

            if (!cap.read(frame)) {
                break;
            }

//            cv::imwrite("/home/manu/tmp/1.bmp", frame);

//            cv::resize(frame, frame_r, cv::Size(640, 480));
            cv::cvtColor(frame, frame_r, cv::COLOR_BGR2RGB);
            cv::cvtColor(frame_r, gray_o, cv::COLOR_RGB2GRAY);
            cv::resize(gray_o, gray, cv::Size(640, 480));
//            gray = letterbox(gray);
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
