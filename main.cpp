#include "smoke_merged_detect_malf.h"

int main() {
    std::vector<std::string> video_paths = {"/home/manu/tmp/vlc-record-2024-11-23-18h14m59s-rtsp___192.168.1.101_visi_stream-.mp4"};
    SmokeMergedDetectMalf md(75, 5, 25, false, true, false, true);
    md.run(video_paths);

    return 0;
}
