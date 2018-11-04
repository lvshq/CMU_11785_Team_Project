echo "Convert train pictures..."
find ../data/Clothes/train_imgs -name '*.jpg' \
	-exec convert -resize 150x150 {} {} \;

echo "Convert test pictures..."
find ../data/Clothes/test_imgs -name '*.jpg' \
	-exec convert -resize 150x150 {} {} \;