set -v
python source/evaluate.py --results results/nefertiti --reference meshes/nefertiti/target.obj
python source/evaluate.py --results results/lucy --reference meshes/lucy/target.obj --alt meshes/lucy/target_normals.obj
python source/evaluate.py --results results/einstein --reference meshes/einstein/target.obj --alt meshes/einstein/target_normals.obj
python source/evaluate.py --results results/metatron --reference meshes/metatron/target.obj --alt meshes/metatron/target_normals.obj
python source/evaluate.py --results results/dragon --reference meshes/dragon/target.obj --alt meshes/dragon/target_normals.obj
python source/evaluate.py --results results/skull --reference meshes/skull/target.obj --alt meshes/skull/target_normals.obj
python source/evaluate.py --results results/xyz --reference meshes/xyz/target.obj
python source/evaluate.py --results results/armadillo --reference meshes/armadillo/target.obj --alt meshes/armadillo/target_normals.obj
python source/evaluate.py --results results/venus --reference meshes/venus/target.obj --alt meshes/venus/target_normals.obj
