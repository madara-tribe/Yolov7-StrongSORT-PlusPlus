#/bin/sh
find . -name '.DS_Store' -type f -ls -delete
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
rm -r *.jpg runs
mkdir -p bp/AFLinks/MOT20 bp/yolov7/models bp/yolov7/utils bp/data bp/weights bp/strong_sort/utils bp/strong_sort/sort bp/strong_sort/deep bp/strong_sort/configs

cp *.py *.txt *.sh Makefile bp/
cp AFLinks/*.py bp/AFLinks/
cp AFLinks/MOT20/*.pth bp/AFLinks/MOT20/
cp yolov7/models/*.py bp/yolov7/models/
cp -r yolov7/utils/* bp/yolov7/utils/
touch bp/data/live_camera.mp4
touch bp/weights/osnet_x0_25_msmt17.pt bp/weights/yolov7-tiny.pt
cp -r strong_sort/*.py bp/strong_sort/
cp -r strong_sort/utils/*.py bp/strong_sort/utils/
cp -r strong_sort/deep/* bp/strong_sort/deep/
cp -r strong_sort/sort/*.py bp/strong_sort/sort/
cp -r strong_sort/configs/* bp/strong_sort/configs/
