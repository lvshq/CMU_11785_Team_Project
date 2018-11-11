# /usr/bin/env sh
TRAIN_DATA=/media/lvshq/LSQ_16GB/SSDH/windows_imgs
LISTFILE=/media/lvshq/LSQ_16GB/SSDH
echo "Create train.txt..."
rm -rf $LISTFILE/pic_for_windows.txt
find $TRAIN_DATA -name "*.jpg" | cut -d '/' -f 7  >> $LISTFILE/pic_for_windows.txt
#rm -rf $DATA/tmp.txt
echo "Done.."