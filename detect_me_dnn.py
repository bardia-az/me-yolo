# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models.common import Bottleneck

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, InterPrediction_1, InterPrediction_2, InterPrediction_3
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadImages_me
from utils.general import (LOGGER, apply_classifier, check_file, check_img_size, check_imshow, check_requirements,
                           check_suffix, colorstr, increment_path, non_max_suppression, print_args, save_one_box,
                           scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        list_path=None,
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        autoenc_chs=None,   # number of channels in the auto encoder
        supp_weights=None,  # autoencoder model.pt path(s)
        weights_me=None,     # motion estimator model.pt path(s)
        deform_G=8,
        model_type='Model-1'
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Loading Autoencoder
    assert supp_weights is not None
    assert supp_weights.endswith('.pt')
    autoencoder = AutoEncoder(autoenc_chs).to(device)
    supp_ckpt = torch.load(supp_weights, map_location=device)
    autoencoder.load_state_dict(supp_ckpt['model'])
    print('autoencoder loaded correctly')
    del supp_ckpt
    if half:
        autoencoder.half()
    autoencoder.eval()
    # Loading Inter-Predictor        
    if model_type == 'Model-1':
        motion_estimator = InterPrediction_1(opt.autoenc_chs[-1]).to(device)
    elif model_type == 'Model-2':
        motion_estimator = InterPrediction_2(opt.autoenc_chs[-1], deform_G).to(device)
    elif model_type == 'Model-3':
        motion_estimator = InterPrediction_3(opt.autoenc_chs[-1], deform_G).to(device)
    else:
        raise Exception(f'model-type={opt.model_type} is not supported')
    assert weights_me is not None
    assert weights_me.endswith('.pt')
    me_ckpt = torch.load(weights_me, map_location=device)
    motion_estimator.load_state_dict(me_ckpt['model'])
    print('motion estimator loaded correctly')
    if half:
        motion_estimator.half()
    motion_estimator.eval()


    dataset = LoadImages_me(list_path, img_size=imgsz, stride=stride)

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, ref1, ref2, target, target_img0, s in dataset:
        t1 = time_sync()
        ref1 = torch.from_numpy(ref1).to(device)
        ref1 = ref1.half() if half else ref1.float()  # uint8 to fp16/32
        ref1 /= 255  # 0 - 255 to 0.0 - 1.0
        ref2 = torch.from_numpy(ref2).to(device)
        ref2 = ref2.half() if half else ref2.float()  # uint8 to fp16/32
        ref2 /= 255  # 0 - 255 to 0.0 - 1.0
        target = torch.from_numpy(target).to(device)
        target = target.half() if half else target.float()  # uint8 to fp16/32
        target /= 255  # 0 - 255 to 0.0 - 1.0
        if len(ref1.shape) == 3:
            ref1 = ref1[None]  # expand for batch dim
        if len(ref2.shape) == 3:
            ref2 = ref2[None]  # expand for batch dim
        if len(target.shape) == 3:
            target = target[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if visualize:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True)
            visualize = visualize / 'features'
            visualize.mkdir(parents=True, exist_ok=True)  # make directory
            visualize_pred = visualize / 'pred'
            visualize_gt = visualize / 'gt'
            visualize_motions = visualize / 'motions'
            visualize_pred.mkdir(parents=True, exist_ok=True)
            visualize_gt.mkdir(parents=True, exist_ok=True)
            visualize_motions.mkdir(parents=True, exist_ok=True)
        else:
            visualize_pred, visualize_gt = False, False

        T1 = model(ref1, augment=augment, cut_model=1, visualize=False)  # first half of the model
        T2 = model(ref2, augment=augment, cut_model=1, visualize=False)  # first half of the model
        Ttrg = model(target, augment=augment, cut_model=1, visualize=visualize)  # first half of the model
        T1_tilde = autoencoder(T1, visualize=visualize, task='enc', s='frame1')
        T2_tilde = autoencoder(T2, visualize=visualize, task='enc', s='frame2')
        Ttrg_hat = motion_estimator(T1_tilde, T2_tilde, visualize=visualize_motions)
        Ttrg_bar_pred = autoencoder(Ttrg, visualize=visualize, task='dec', bottleneck=Ttrg_hat, s='pred')
        Ttrg_bar = autoencoder(Ttrg, visualize=visualize, task='enc_dec', s='gt')

        pred_pred = model(None, cut_model=2, T=Ttrg_bar_pred, visualize=visualize_pred)[0]  # second half of the model  
        pred = model(None, cut_model=2, T=Ttrg_bar, visualize=visualize_gt)[0]  # second half of the model  
    
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred_pred = non_max_suppression(pred_pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        img = target
        im0s = target_img0
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), 0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name / 'objs_original.jpg')  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)


        for i, det in enumerate(pred_pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), 0

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name / 'objs_pred.jpg')  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    # Supplemental arguments
    parser.add_argument('--autoenc-chs',  type=int, nargs='*', default=[320,192,64], help='number of channels in autoencoder')
    parser.add_argument('--supp-weights', type=str, default=None, help='initial weights path for the autoencoder')
    parser.add_argument('--weights-me', type=str, default=None, help='initial weights path for the motion estimator')
    parser.add_argument('--list-path', type=str, default=None, help='txt file containing the paths list')
    parser.add_argument('--deform-G', type=int, default=8, help='number of groups in deformable convolution layers')
    parser.add_argument('--model-type', type=str, default='Model-1', help='Which Inter-Prediction Model? Model-1, Model-2, or Model-3')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
