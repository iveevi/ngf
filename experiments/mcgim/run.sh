# set -e
# for i in {10..250..10}
# do
# 	echo "### Running construction with $i patches at 16 x 16 resolution"
# 	python construct.py ../../meshes/nefertiti/target.obj $i 16
# done

for filename in results/nefertiti/mcgim*.pt
do
	echo "Training neural representation of $filename"
	python train.py --mcgim $filename
done
