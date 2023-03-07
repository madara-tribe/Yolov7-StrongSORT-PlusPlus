run:
	python3 track_v7.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --save-txt --count --show-vid --draw

link:
	python3 track_v7.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --save-txt --count --show-vid --draw --post

post:
	python3 post_track.py --AFLink --GSI

post_plot:
	python3 post_track.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --show-vid --draw --post_plot

