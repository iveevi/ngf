set -e -v
git submodule update --init
cmake -B build .
cmake --build build
gdown --id 10qhu9uPgxtvnObGamTVCB0y_yQ_29ps0
tar -xzvf meshes.tar.gz
rm meshes.tar.gz
