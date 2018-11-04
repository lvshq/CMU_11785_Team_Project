echo "Create train mean.."
../build/tools/compute_image_mean \
../data/Clothes/img_train_lmdb \
../data/Clothes/train_mean.binaryproto

echo "Create test mean.."
../build/tools/compute_image_mean \
../data/Clothes/img_test_lmdb \
../data/Clothes/test_mean.binaryproto