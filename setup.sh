#/bin/bash
set -e
git submodule update --init
cmake -B build .
cmake --build build
gdown --id 10qhu9uPgxtvnObGamTVCB0y_yQ_29ps0
