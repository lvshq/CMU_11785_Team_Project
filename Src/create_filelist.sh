# /usr/bin/env sh
TRAIN_DATA=../data/Clothes/train_imgs
TEST_DATA=../data/Clothes/test_imgs
LISTFILE=../data/Clothes
echo "Create train.txt..."
rm -rf $LISTFILE/train.txt
rm -rf $LISTFILE/test.txt

find $TRAIN_DATA -name Coat_Male_*.jpg | cut -d '/' -f 5 | sed "s/$/ 0/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Coat_Female_*.jpg | cut -d '/' -f 5 | sed "s/$/ 1/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Tshirt_Male_*.jpg | cut -d '/' -f 5 | sed "s/$/ 2/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Tshirt_Female_*.jpg | cut -d '/' -f 5 | sed "s/$/ 3/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Pants_*.jpg | cut -d '/' -f 5 | sed "s/$/ 4/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Down_Jacket_*.jpg | cut -d '/' -f 5 | sed "s/$/ 5/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Sweater_*.jpg | cut -d '/' -f 5 | sed "s/$/ 6/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Vest_*.jpg | cut -d '/' -f 5 | sed "s/$/ 7/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Suit_*.jpg | cut -d '/' -f 5 | sed "s/$/ 8/">>$LISTFILE/train.txt
find $TRAIN_DATA -name Dress_*.jpg | cut -d '/' -f 5 | sed "s/$/ 9/">>$LISTFILE/train.txt
echo "Create test.txt..."

find $TEST_DATA -name Coat_Male_*.jpg | cut -d '/' -f 5 | sed "s/$/ 0/">>$LISTFILE/test.txt
find $TEST_DATA -name Coat_Female_*.jpg | cut -d '/' -f 5 | sed "s/$/ 1/">>$LISTFILE/test.txt
find $TEST_DATA -name Tshirt_Male_*.jpg | cut -d '/' -f 5 | sed "s/$/ 2/">>$LISTFILE/test.txt
find $TEST_DATA -name Tshirt_Female_*.jpg | cut -d '/' -f 5 | sed "s/$/ 3/">>$LISTFILE/test.txt
find $TEST_DATA -name Pants_*.jpg | cut -d '/' -f 5 | sed "s/$/ 4/">>$LISTFILE/test.txt
find $TEST_DATA -name Down_Jacket_*.jpg | cut -d '/' -f 5 | sed "s/$/ 5/">>$LISTFILE/test.txt
find $TEST_DATA -name Sweater_*.jpg | cut -d '/' -f 5 | sed "s/$/ 6/">>$LISTFILE/test.txt
find $TEST_DATA -name Vest_*.jpg | cut -d '/' -f 5 | sed "s/$/ 7/">>$LISTFILE/test.txt
find $TEST_DATA -name Suit_*.jpg | cut -d '/' -f 5 | sed "s/$/ 8/">>$LISTFILE/test.txt
find $TEST_DATA -name Dress_*.jpg | cut -d '/' -f 5 | sed "s/$/ 9/">>$LISTFILE/test.txt

awk 'BEGIN{ 100000*srand();}{ printf "%s %s\n", rand(), $0}' $LISTFILE/train.txt |
 sort -k1n | awk '{gsub($1FS,""); print $0}' > $LISTFILE/train_unorder.txt

awk 'BEGIN{ 100000*srand();}{ printf "%s %s\n", rand(), $0}' $LISTFILE/test.txt |
 sort -k1n | awk '{gsub($1FS,""); print $0}' > $LISTFILE/test_unorder.txt

#cat $DATA/tmp.txt>>$DATA/train.txt
#rm -rf $DATA/tmp.txt
echo "Done.."