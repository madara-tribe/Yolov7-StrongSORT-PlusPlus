import os
import sys
import argparse
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from random import random as ran
from datetime import datetime
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, apply_classifier, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from track_utils import img_preprocess, rescale_bbox, process_results, get_boxes_result, inference_with_nms


from AFLinks.AppFreeLink import AFlink
from GSI import GSInterpolation

def main(opt):
    if opt.AFLink:
        aflink = AFlink(folder=opt.folder,
                        inpath=opt.inpath,
                        outpath=opt.outpath,
                        model_path=opt.model_path)
        res = aflink.link(save=True)
        print("aflink", res.shape)
    if opt.GSI:
        GSInterpolation(path_in=os.path.join(opt.folder, opt.outpath),
                        path_out=os.path.join(opt.folder, opt.gsi_outpath),
                        interval=20,
                        tau=10,
                        save=True)
    if opt.post_plot:
        detect(opt)
                        
      
def detect(opt, save_img=False, line_thickness=1):
    source, weights, show_vid, imgsz, trace = opt.source, opt.yolo_weights, opt.show_vid, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_crop = False
    project=ROOT / 'runs/track'  # save results to project/name
    exp_name='exp'  # save results to project/name
    save_vid=opt.save_vid
    save_img=opt.save_img
    line_thickness=opt.line_thickness
    draw=opt.draw
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_vid else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    trajectory = {}
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    
    # load post process track txt
    post_tracks = np.loadtxt(os.path.join(opt.folder, opt.gsi_outpath), delimiter=',')
    # for path, img, im0s, vid_cap in dataset:
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        t1 = time_synchronized()
        img = img_preprocess(img, device, half)

        # Inference and Apply NMS
        t2 = time_synchronized()
        dt[0] += t2 - t1
        pred, dt, t3, t2 = inference_with_nms(model, img, opt, dt, t2)
        dt[2] += time_synchronized() - t3
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)


            curr_frames[i] = im0
            p = Path(p)  # to Path
            txt_file_name = p.name
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, .
            txt_path = str(save_dir / 'labels' / p.stem)  # im.txt

            s += '%gx%g ' % img.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
        
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            res = post_tracks[post_tracks[:, 0]==frame_idx+1]
            
            if len(det) and len(res):
                # Rescale boxes from img_size to im0 size
                det = rescale_bbox(img, det, im0)

                # Print results
                xywhs, confs, clss, s = process_results(det, s, names)

                # draw boxes for visualization
                for j, rows in enumerate(res):
                    bboxes, id, cls = rows[2:6], rows[1], 1
                    if draw:
                        center = ((int(bboxes[0]) + int(bboxes[2])) // 2,(int(bboxes[1]) + int(bboxes[3])) // 2)
                        if id not in trajectory:
                            trajectory[id] = []
                        trajectory[id].append(center)
                        for i1 in range(1,len(trajectory[id])):
                            if trajectory[id][i1-1] is None or trajectory[id][i1] is None:
                                continue
                            # thickness = int(np.sqrt(1000/float(i1+10))*0.3)
                            thickness = 2
                            try:
                              cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255), thickness)
                            except:
                              pass

                    if save_vid or save_crop or show_vid :  # Add bbox to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = f'{id} {names[c]}'
                        plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=line_thickness)

                ### Print time (inference + NMS)
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s)')

            else:
                print('No detections')

            #current frame // tesing
            cv2.imwrite('post_track.jpg',im0)

            # Stream results
            if show_vid:
                inf = (f'{s}Done. ({t2 - t1:.3f}s)')
                # cv2.putText(im0, str(inf), (30,160), cv2.FONT_HERSHEY_SIMPLEX,0.7,(40,40,40),2)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

            prev_frames[i] = curr_frames[i]
    
    if save_vid or save_img:
        print(f"Results saved to ",save_dir)
    print(f'Done. ({time.time() - t0:.3f}s)')
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default='weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',default=True, help='display results')
    parser.add_argument('--save-img', action='store_true', help='save results to *.jpg')
    parser.add_argument('--nosave', action='store_true',default=True, help='do not save images/videos')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--exp-name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    
    # post-tracking
    parser.add_argument('--folder', type=str, default='runs/track/exp/labels', help='text folder')
    parser.add_argument('--inpath', type=str, default='live_camera_aflink.txt', help='text path to put in')
    parser.add_argument('--outpath', type=str, default='post_aflink.txt', help='aflink text path to put out')
    parser.add_argument('--gsi_outpath', type=str, default='post_gsi.txt', help='gsi text path to put out')
    parser.add_argument('--model_path', type=str, default='AFlinks/MOT20/AFLink_epoch20.pth', help='AFLink model path')
    parser.add_argument('--AFLink', action='store_true', help='Appearance-Free Link')
    parser.add_argument('--GSI', action='store_true', help='Gaussian-smoothed Interpolation')
    parser.add_argument('--post_plot', action='store_true', help='plot post processing tracking on frames')
    opt = parser.parse_args()
    main(opt)
    

            

