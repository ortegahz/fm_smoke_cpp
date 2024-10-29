#include "smoke_merged_detect_malf.h"

int main() {
    std::vector<std::string> video_paths = {"/media/manu/ST8000DM004-2U91/smoke/fault/vas2.mp4"};
    SmokeMergedDetectMalf md(75, 5, 25, false, true, false, true);
    md.run(video_paths);

    return 0;
}
