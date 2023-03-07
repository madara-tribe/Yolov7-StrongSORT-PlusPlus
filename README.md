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

<img src="https://user-images.githubusercontent.com/48679574/223443273-ec6b5737-e5e0-44e6-bc1b-52d0bc6e28d7.jpg" width="600" height="450"/>

## StrongSORT++ (with AFlink and GSI postprosecing)

<img src="https://user-images.githubusercontent.com/48679574/223443233-e1d41f0f-a094-4d70-b70e-ed891991a986.jpg" width="600" height="450"/>


# References
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [StrongSORT-YOLO](https://github.com/bharath5673/StrongSORT-YOLO)
