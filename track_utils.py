from strong_sort.strong_sort import StrongSORT
import torch
from yolov7.utils.general import scale_coords, xyxy2xywh, non_max_suppression
from yolov7.utils.torch_utils import time_synchronized

def strongsort_instances(nr_sources, strong_sort_weights, device, cfg):
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    return strongsort_list


def img_preprocess(img, device, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def rescale_bbox(img, det, im0):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    return det

def process_results(det, s, names):
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    xywhs = xyxy2xywh(det[:, 0:4])
    confs = det[:, 4]
    clss = det[:, 5]
    return xywhs, confs, clss, s

def get_boxes_result(output):
    bboxes = output[0:4]
    id = output[4]
    cls = output[5]
    return bboxes, id, cls


        
def inference_with_nms(model, img, opt, dt, t2):
    pred = model(img, augment=opt.augment)[0]
    t3 = time_synchronized()
    dt[1] += t3 - t2
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    return pred, dt, t3, t2
