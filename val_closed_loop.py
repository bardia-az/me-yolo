# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
from enum import auto
import json
import os
import sys
from pathlib import Path
from threading import Thread
import yaml

import numpy as np
import torch
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, InterPrediction_1, InterPrediction_2, InterPrediction_3
from utils.datasets import create_dataloader
from utils.general import (LOGGER, StatCalculator, check_file, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, print_args_to_file)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def read_y_channel(file, w, h):
    raw = file.read(w*h)
    raw = np.frombuffer(raw, dtype=np.uint8)
    raw = raw.reshape((h,w))
    return raw

def get_tensors(img, model, autoencoder):
    T = model(img, cut_model=1)  # first half of the model
    if autoencoder is not None:
        T_bottleneck = autoencoder(T, task='enc')
    else:
        T_bottleneck = T
    return T_bottleneck

def motion_estimation(ref_tensors, model):
    if model is not None:
        _, _, ch_h, ch_w = ref_tensors.shape
        T_in = ref_tensors.detach().clone()
        return model(T_in[0, ...].unsqueeze(0), T_in[1, ...].unsqueeze(0))
    else:
        return torch.zeros(1, device=ref_tensors.device)

def tensors_to_tiled(tensor, tensors_w, tensors_h, tensors_min, tensors_max):
    shape = tensor.shape
    ch_num_w, ch_num_h = tensors_w // shape[3], tensors_h // shape[2]
    assert shape[1] == ch_num_w*ch_num_h, 'tensors_w & tensors_h are not valid values based on the latent tensors'
    tensor = torch.clamp(tensor, tensors_min, tensors_max)
    tensor = torch.reshape(tensor, (ch_num_h, ch_num_w, shape[2], shape[3]))
    tensor = torch.round((tensor - tensors_min) * 255 / (tensors_max - tensors_min))
    return tensor.swapaxes(1,2).reshape((tensors_h, tensors_w))

def tiled_to_tensor(tiled, ch_w, ch_h, tensors_min, tensors_max):
    shape = tiled.shape
    ch_num_w, ch_num_h = shape[1] // ch_w, shape[0] // ch_h
    tiled = torch.reshape(tiled, (ch_num_h, ch_h, ch_num_w, ch_w))
    tiled = tiled / 255 * (tensors_max - tensors_min) + tensors_min
    tensor = tiled.swapaxes(1,2).reshape((-1, ch_h, ch_w))
    return tensor

def encode_frame(data, tensors_w, tensors_h, txt_file, frame_rate, qp):
    VVC_command = ['../vvc/vvencFFapp', '-c', '../vvc/lowdelay_faster_latent.cfg', '-i', '../vvc/to_be_coded_frame.yuv', '-b', '../vvc/bitstream.bin', 
                   '-o', '../vvc/reconst.yuv', '--SourceWidth', str(tensors_w), '--SourceHeight', str(tensors_h), '-f', '1', '-fr', str(frame_rate), '-q', str(qp)]
    to_be_coded_file = '../vvc/to_be_coded_frame.yuv'
    with open(to_be_coded_file, 'wb') as f:
        f.write(data)
    subprocess.call(VVC_command, stdout=txt_file)
    Byte_num = os.path.getsize('../vvc/bitstream.bin')/1024.0
    with open('../vvc/reconst.yuv', 'rb') as f:
        tmp_reconst = read_y_channel(f, tensors_w, tensors_h)
    return tmp_reconst, Byte_num

def get_yolo_prediction(T, autoencoder, model):
    if autoencoder is not None:
        T_hat = autoencoder(T, task='dec', bottleneck=T)
        out, _ = model(None, cut_model=2, T=T_hat)  # second half of the model
    else:
        out, _ = model(None, cut_model=2, T=T)  # second half of the model
    return out



@torch.no_grad()
def val_closed_loop(opt,
                    callbacks=Callbacks()):

    save_dir, batch_size, weights, single_cls, data, supp_weights, half, weights_me, plots, tensors_w, tensors_h, tensors_min, \
        tensors_max, res_min, res_max, save_videos, track_stats, lossless, tensor_video, save_original = \
        Path(opt.save_dir), opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.supp_weights, opt.half, opt.weights_me, False, opt.tensors_w, \
        opt.tensors_h, opt.tensors_min, opt.tensors_max, opt.res_min, opt.res_max, opt.save_videos, opt.track_stats, opt.lossless, opt.tensor_video, opt.save_original

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    device = select_device(opt.device, batch_size=batch_size)

    save_videos &= not(lossless)
    track_stats &= not(lossless)

    # Load model
    check_suffix(weights, '.pt')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, s=gs)  # check image size

    # Data
    video_name = data.split('/')[-1].split('.')[0]
    video = f'{video_name}_QP{opt.qp}'
    data = check_dataset(data, suffix=opt.data_suffix)  # check
    if tensor_video is not None:
        tensor_video = Path(tensor_video)
        tensor_video_vvc = (tensor_video / video_name).with_suffix('.bin')
        tensor_video = (tensor_video / video_name).with_suffix('.yuv')

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Loading Autoencoder and Motion Estimator
    autoencoder = None
    if supp_weights is not None:
        assert supp_weights.endswith('.pt'), 'autoencoder weight file format not supported ".pt"'
        autoencoder = AutoEncoder(opt.autoenc_chs).to(device)
        supp_ckpt = torch.load(supp_weights, map_location=device)
        autoencoder.load_state_dict(supp_ckpt['model'])
        print('autoencoder loaded successfully')
        del supp_ckpt
        autoencoder.half() if half else autoencoder.float()
        autoencoder.eval()

    motion_estimator = None
    if weights_me is not None:
        assert weights_me.endswith('.pt'), 'motion estimator weight file format not supported ".pt"'
        if opt.model_type == 'Model-1':
            motion_estimator = InterPrediction_1(opt.autoenc_chs[-1]).to(device)
        elif opt.model_type == 'Model-2':
            motion_estimator = InterPrediction_2(opt.autoenc_chs[-1], opt.deform_G).to(device)
        elif opt.model_type == 'Model-3':
            motion_estimator = InterPrediction_3(opt.autoenc_chs[-1], opt.deform_G).to(device)
        else:
            raise Exception(f'model-type={opt.model_type} is not supported')
        me_ckpt = torch.load(weights_me, map_location=device)
        motion_estimator.load_state_dict(me_ckpt['model'])
        print('motion estimator loaded successfully')
        del me_ckpt
        motion_estimator.half() if half else motion_estimator.float()
        motion_estimator.eval()
    else:
        res_min = tensors_min
        res_max = tensors_max

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    pad = 0.0
    task = 'val'
    video_sequence = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=False, rect_img=False,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    stats_error = StatCalculator(opt.dist_range, opt.bins) if track_stats else None

    if save_videos and tensor_video is None:
        (save_dir / 'to_be_coded').mkdir(parents=True, exist_ok=True)
        (save_dir / 'reconstructed').mkdir(parents=True, exist_ok=True)
        (save_dir / 'original').mkdir(parents=True, exist_ok=True)
        if motion_estimator is not None:
            (save_dir / 'predicted').mkdir(parents=True, exist_ok=True)
        to_be_coded_full_name = (save_dir / 'to_be_coded' / f'to_be_coded_{video}').with_suffix('.yuv')
        predicted_full_name = (save_dir / 'predicted' / f'predicted_{video}').with_suffix('.yuv')
        reconstructed_full_name = (save_dir / 'reconstructed' / f'reconstructed_{video}').with_suffix('.yuv')
        original_full_name = (save_dir / 'original' / f'original_{video}').with_suffix('.yuv')
    if save_videos:
        (save_dir / 'error').mkdir(parents=True, exist_ok=True)
        error_full_name = (save_dir / 'error' / f'error_{video}').with_suffix('.yuv')

    if track_stats:
        (save_dir / 'error_stats').mkdir(parents=True, exist_ok=True)
        error_dir = (save_dir / 'error_stats')

    if tensor_video is None and not(lossless):
        (save_dir / 'report').mkdir(parents=True, exist_ok=True)
        (save_dir / 'plots').mkdir(parents=True, exist_ok=True)
        report_file_name = (save_dir / 'report' / video).with_suffix('.txt')
        plot_dir = (save_dir / 'plots')

    if save_videos:
        assert not (error_full_name.exists()), 'This run has been done before. Videos are already available. Delete the corresponding tiled tensor videos to run again.'

    if tensor_video is None:
        full_to_be_coded_frame_f = to_be_coded_full_name.open('ab') if save_videos else None
        full_predicted_f = predicted_full_name.open('ab') if save_videos and motion_estimator is not None else None
        reconst_video_f = reconstructed_full_name.open('ab') if save_videos else None
        original_full_f = original_full_name.open('ab') if save_videos else None
        report_file = report_file_name.open('a') if not(lossless) else None
    tensor_video_f = Path(tensor_video).open('rb') if tensor_video is not None else None
    error_full_f = error_full_name.open('ab') if save_videos else None

    if save_original:
        (save_dir / 'tiled_tensors').mkdir(parents=True, exist_ok=True)
        original_full_name = (save_dir / 'tiled_tensors' / f'latent_{video_name}').with_suffix('.yuv')
        original_full_f = original_full_name.open('ab')

    optimal_qp_for_I_frames = 28
    ref_num = 2
    ch_w, ch_h = opt.ch_dim, opt.ch_dim
    ch_num_w, ch_num_h = (tensors_w//ch_w), (tensors_h//ch_h)
    ch_num = ch_num_w * ch_num_h
    ref_tensors = torch.zeros((ref_num, ch_num, ch_w, ch_h), device=device, dtype=torch.half if half else torch.float)
    Byte_num_seq = []

    for fr, (img, targets, paths, shapes) in enumerate(tqdm(video_sequence, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        tensors = get_tensors(img, model, autoencoder)
        if tensor_video is None:
            if not lossless:
                if fr < ref_num:
                    tiled_tensors = tensors_to_tiled(tensors, tensors_w, tensors_h, tensors_min, tensors_max)
                    to_be_coded_frame_data = tiled_tensors.cpu().numpy().flatten().astype(np.uint8)
                    if save_videos:
                        full_to_be_coded_frame_f.write(to_be_coded_frame_data)
                        original_full_f.write(to_be_coded_frame_data)
                    tmp_reconst, Byte_num = encode_frame(to_be_coded_frame_data, tensors_w, tensors_h, report_file, data['frame_rate'], opt.qp if opt.qp < optimal_qp_for_I_frames or motion_estimator is None else optimal_qp_for_I_frames)
                    Byte_num_seq.append(Byte_num)
                    if save_videos:
                        reconst_video_f.write(tmp_reconst.flatten())
                    tiled_tensors = torch.from_numpy(tmp_reconst.copy()).to(device)
                    tiled_tensors = tiled_tensors.half() if half else tiled_tensors.float()
                    rec_tensors = tiled_to_tensor(tiled_tensors, ch_w, ch_h, tensors_min, tensors_max)
                    ref_tensors[fr,:,:,:] = rec_tensors
                else:
                    pred_tensors = motion_estimation(ref_tensors, motion_estimator)
                    residual = tensors - pred_tensors
                    tiled_res = tensors_to_tiled(residual, tensors_w, tensors_h, res_min, res_max)
                    to_be_coded_frame_data = tiled_res.cpu().numpy().flatten().astype(np.uint8)
                    if save_videos:
                        tiled_orig = tensors_to_tiled(tensors, tensors_w, tensors_h, tensors_min, tensors_max)
                        orig_data = tiled_orig.cpu().numpy().flatten().astype(np.uint8)
                        original_full_f.write(orig_data)
                        full_to_be_coded_frame_f.write(to_be_coded_frame_data)
                    tmp_reconst, Byte_num = encode_frame(to_be_coded_frame_data, tensors_w, tensors_h, report_file, data['frame_rate'], opt.qp)
                    Byte_num_seq.append(Byte_num)
                    tiled_res = torch.from_numpy(tmp_reconst.copy()).to(device)
                    tiled_res = tiled_res.half() if half else tiled_res.float()
                    residual = tiled_to_tensor(tiled_res, ch_w, ch_h, res_min, res_max)
                    rec_tensors = residual + pred_tensors[0]
                    ref_tensors = torch.roll(ref_tensors, -1, 0)
                    ref_tensors[ref_num-1,:,:,:] = rec_tensors
                    
                    # # for logging video 
                    if save_videos:
                        tiled_tensors = tensors_to_tiled(rec_tensors[None, :], tensors_w, tensors_h, tensors_min, tensors_max)
                        reconst_video_f.write(tiled_tensors.cpu().numpy().flatten().astype(np.uint8))
                        if motion_estimator is not None:
                            tiled_pred = tensors_to_tiled(pred_tensors, tensors_w, tensors_h, tensors_min, tensors_max)
                            pred_data = tiled_pred.cpu().numpy().flatten().astype(np.uint8)
                            full_predicted_f.write(pred_data)
        else:
            tmp_reconst = read_y_channel(tensor_video_f, tensors_w, tensors_h)
            tiled_tensors = torch.from_numpy(tmp_reconst.copy()).to(device)
            rec_tensors = tiled_to_tensor(tiled_tensors, ch_w, ch_h, tensors_min, tensors_max)

        if lossless:
            out = get_yolo_prediction(tensors, autoencoder, model)
            if save_original:
                tiled_tensors = tensors_to_tiled(tensors, tensors_w, tensors_h, tensors_min, tensors_max)
                to_be_coded_frame_data = tiled_tensors.cpu().numpy().flatten().astype(np.uint8)
                original_full_f.write(to_be_coded_frame_data)
        else:
            out = get_yolo_prediction(rec_tensors[None, :], autoencoder, model)
            error = rec_tensors - tensors[0]
            if track_stats:
                stats_error.update_stats(error.detach().clone().cpu().numpy())
            if save_videos:
                tiled_error = tensors_to_tiled(error[None, :], tensors_w, tensors_h, res_min, res_max)
                error_data = tiled_error.cpu().numpy().flatten().astype(np.uint8)
                error_full_f.write(error_data)


        dt[1] += time_sync() - t2

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if opt.save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, opt.conf_thres, opt.iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if opt.save_txt:
                save_one_txt(predn, opt.save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if opt.save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and fr < 3:
            f = save_dir / f'val_batch{fr}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{fr}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    if save_videos and tensor_video is None:
        full_to_be_coded_frame_f.close()
        reconst_video_f.close()
        original_full_f.close()
        if motion_estimator is not None:
            full_predicted_f.close()
    if save_videos:
        error_full_f.close()
    if tensor_video is not None:
        tensor_video_f.close()
    if save_original:
        original_full_f.close()
    
    # Compute statistics
    if opt.res_per_frame:
        map50_fr, map_fr = [], []
        map50_tmp, map_tmp = 0.0, 0.0
        for stat_tmp in stats:
            stat = [np.array(x) if isinstance(x, list) else x.numpy() for x in stat_tmp]  # to numpy
            if len(stat) and stat[0].any():
                _, _, ap_tmp, _, _ = ap_per_class(*stat, plot=plots, save_dir=save_dir, names=names)
                ap50_tmp, ap_tmp = ap_tmp[:, 0], ap_tmp.mean(1)  # AP@0.5, AP@0.5:0.95
                map50_tmp, map_tmp = ap50_tmp.mean(), ap_tmp.mean()
                nt = np.bincount(stat[3].astype(np.int64), minlength=nc)  # number of targets per class
            else:
                nt = torch.zeros(1)
            map50_fr.append(map50_tmp)
            map_fr.append(map_tmp)
        file = (save_dir / f'result_frame_{video_name}').with_suffix('.csv')
        with open(file, 'a') as f:
            np.savetxt(file, map50_fr, fmt='%20.5g')
            
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    if tensor_video is not None:
        Byte_num = os.path.getsize(tensor_video_vvc)/1024.0
        Byte_num_seq.append(Byte_num)
    bit_rate = sum(Byte_num_seq) * 8 * data['frame_rate'] / len(video_sequence)

    # Print results   
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # Save results
    file = (save_dir / f'result_{video_name}').with_suffix('.csv')
    s = '' if file.exists() else (('%20s,' * 6) % ('QP', 'Bit-rate (kb/s)', 'mAP@.5 (%)', 'mAP@.5:.95 (%)', 'P', 'R')) + '\n'  # add header
    with open(file, 'a') as f:
        f.write(s + (('%20.5g,' * 6) % (opt.qp, bit_rate, map50*100, map*100, mp, mr)) + '\n')
    
    if track_stats:
        stats_error.output_stats(error_dir, video)
        
    # Print results per class
    if (opt.verbose or nc<50) and nc > 1 and len(stats):
        file = (save_dir / f'res_per_class_{video_name}').with_suffix('.csv')
        s = '' if file.exists() else (('%20s,' * 8) % ('QP', 'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')) + '\n'  # add header
        with open(file, 'a') as f:
            f.write(s + (('%20.5g,' + '%20s,' + '%20.5g,' * 6) % (opt.qp, 'all', seen, nt.sum(), mp, mr, map50, map)) + '\n')
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                f.write(('%20.5g,' + '%20s,' + '%20.5g,' * 6) % (opt.qp, names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')

    if tensor_video is None and not(lossless):
        plt.plot(Byte_num_seq)
        plt.xlabel('frame number')
        plt.ylabel('frame size (kB)')
        plt.title(video)
        plt.savefig(str(plot_dir / f'rate_fig_{video}.png'))
    print(f'\nbit rate (kb/s) <{video}> = {bit_rate:.2f}')
    print(f'size of the compressed file (kB) <{video}> = {sum(Byte_num_seq):.2f}\n')

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if opt.save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in video_sequence.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}\n")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(video_sequence)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/closed_loop', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    parser.add_argument('--add-noise', action='store_true', help='adding noise to the cutting point')
    parser.add_argument('--noise-type', default='uniform', choices=['uniform', 'gaussian', 'uni_gaussian', 'arbitrary'], help='type of the added noise')
    parser.add_argument('--gauss-var', type=float, default=1, help='variance of the gaussian noise')
    parser.add_argument('--uniform-len', type=float, default=1, help='the length of the uniform distribution')
    parser.add_argument('--arbitrary-dist', default=None, help='the numpy file containing the distribution')

    # Supplemental arguments
    parser.add_argument('--autoenc-chs',  type=int, nargs='*', default=[320, 192, 64], help='number of channels in autoencoder')
    parser.add_argument('--supp-weights', type=str, default=None, help='initial weights path for the autoencoder')
    parser.add_argument('--weights-me', type=str, default=None, help='initial weights path for the autoencoder')
    parser.add_argument('--track-stats', action='store_true', help='track the statistical properties of the error')
    parser.add_argument('--dist-range',  type=float, nargs='*', default=[-4,4], help='the range of the distribution')
    parser.add_argument('--bins', type=int, default=10000, help='number of bins in histogram')
    parser.add_argument('--qp', type=int, default=24, help='QP for the vvc encoder')
    parser.add_argument('--tensors-w', type=int, default=1024, help='width of the tiled tensor frames')
    parser.add_argument('--tensors-h', type=int, default=1024, help='height of the tiled tensor frames')
    parser.add_argument('--ch-dim', type=int, default=128, help='height or width of the channels in the latent space')
    parser.add_argument('--tensors-min', type=float, default=-0.2786, help='the clipping lower bound for the intermediate tensors')
    parser.add_argument('--tensors-max', type=float, default=1.4, help='the clipping upper bound for the intermediate tensors')
    parser.add_argument('--res-min', type=float, default=-1, help='the clipping lower bound for the residuals')
    parser.add_argument('--res-max', type=float, default=1, help='the clipping upper bound for the residuals')
    parser.add_argument('--save-videos', action='store_true', help='save the predicted, to-be-coded, and reconstructed videos')
    parser.add_argument('--data-suffix', type=str, default='', help='data path suffix')
    parser.add_argument('--lossless', action='store_true', help='This flag indicates evaluation without compression')
    parser.add_argument('--save-original', action='store_true', help='This flag indicates saving the original latent space video')
    parser.add_argument('--tensor-video', type=str, default=None, help='the path of the tiled tensor video')
    parser.add_argument('--res-per-frame', action='store_true', help='save the mAP results per frame')
    parser.add_argument('--deform-G', type=int, default=8, help='number of groups in deformable convolution layers')
    parser.add_argument('--model-type', type=str, default='Model-1', help='Which Inter-Prediction Model? Model-1, Model-2, or Model-3')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    # Directories
    opt.data, opt.weights, opt.project = check_file(opt.data), str(opt.weights), str(opt.project)  # checks
    save_dir = (Path(opt.project) / opt.name)  # increment run
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    opt.save_dir = str(save_dir)

    val_closed_loop(opt)


def run(**kwargs):
    # Usage: import val_closed_loop; val_closed_loop.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
