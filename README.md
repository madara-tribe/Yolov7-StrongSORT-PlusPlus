# StrongSORT++ with Yolov7 tracking

StrongSORT++ is a more accurate model by applying AFLink (offline processing) and GSI interpolation (post-processing) than StrongSORT.

mainly it can be used for "Traffic analysis", "Realtime Tracking" etc.

this is basic material to improve many models for myself.


# How to use

## preparation
- yolov7-tiny
- movie
- strongsort weight 
- AFlink model weight 

```bash
# just tracking 
($ make run)
$ python3 track_v7.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --save-txt --count --show-vid --draw

# tracking before AFlink and GSI
($ make link)
$ python3 track_v7.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --save-txt --count --show-vid --draw --post

# applay AFLink and GSI after tracking
($ make post)
$ python3 post_track.py --AFLink --GSI

# plot preprocessing (AFlink and GSI) tracking result on movie frames
($make post_plot)
$ python3 post_track.py --source data/live_camera.mp4 --yolo-weights weights/yolov7-tiny.pt --show-vid --draw --post_plot
```


# Performance

## StrongSORT (without AFlink and GSI)



## StrongSORT++ (with AFlink and GSI postprosecing)



# References
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [StrongSORT-YOLO](https://github.com/bharath5673/StrongSORT-YOLO)
