set -e -v

mkdir -p generated

# ./testbed ../../meshes/xyz/target.obj ../../compiled/armadillo/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/armadadillo.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/dragon/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/dragon.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/einstein/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/eintestein.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/lucy/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/lucy.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/metatron/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/metatron.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/nefertiti/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/nefertiti.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/skull/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/skull.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/xyz/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/xyz.mp4 -y
#
# ./testbed ../../meshes/xyz/target.obj ../../compiled/venus/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/venus.mp4 -y

# ./testbed ../../meshes/xyz/target.obj ../../compiled/igea/lod2.nsc
# ffmpeg -framerate 30 -i video/frame%03d.png generated/igea.mp4 -y

./testbed ../../meshes/xyz/target.obj ../../compiled/laplacian/with_laplacian.nsc
ffmpeg -framerate 30 -i video/frame%03d.png generated/indonesian.mp4 -y
