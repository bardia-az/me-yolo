# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from turtle import position

import numpy as np
import torch
from tqdm import tqdm
# import torchjpeg.dct as dct

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.supplemental import AutoEncoder, InterPrediction_1, InterPrediction_2, InterPrediction_3
from utils.datasets import create_me_dataloader
from utils.general import (LOGGER, StatCalculator, box_iou, check_dataset, check_img_size, check_requirements, check_suffix, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, print_args_to_file)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.loss import ComputeLossME


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
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        data_list,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        autoenc_chs=None,
        supp_weights=None,  # model.pt path(s)
        weights_me=None,
        track_stats=False,
        dist_range=[-10,14],
        bins=10000,
        feature_max=20,
        deform_G=8,
        model_type='Model-1',
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss_me=None,
        autoencoder=None,
        motion_estimator=None
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        print_args_to_file(opt, save_dir / 'result.txt')

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        # data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Loading Autoencoder and Motion Estimator
    if not training:
        assert supp_weights is not None, 'autoencoder weights should be available'
        assert supp_weights.endswith('.pt'), 'autoencoder weight file format not supported ".pt"'
        autoencoder = AutoEncoder(autoenc_chs).to(device)
        supp_ckpt = torch.load(supp_weights, map_location=device)
        autoencoder.load_state_dict(supp_ckpt['model'])
        print('autoencoder loaded successfully')
        del supp_ckpt
        autoencoder.half() if half else autoencoder.float()

        assert weights_me is not None, 'motion estimator weights should be available'
        assert weights_me.endswith('.pt'), 'motion estimator weight file format not supported ".pt"'
        if model_type == 'Model-1':
            motion_estimator = InterPrediction_1(opt.autoenc_chs[-1]).to(device)
        elif model_type == 'Model-2':
            motion_estimator = InterPrediction_2(opt.autoenc_chs[-1], deform_G).to(device)
        elif model_type == 'Model-3':
            motion_estimator = InterPrediction_3(opt.autoenc_chs[-1], deform_G).to(device)
        else:
            raise Exception(f'model-type={opt.model_type} is not supported')
        me_ckpt = torch.load(weights_me, map_location=device)
        motion_estimator.load_state_dict(me_ckpt['model'])
        print('motion estimator loaded successfully')
        del me_ckpt
        motion_estimator.half() if half else autoencoder.float()

    # Configure
    model.eval()
    motion_estimator.eval()
    autoencoder.eval()
    # is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_me_dataloader(data_list, imgsz, batch_size, rank=-1, augment=False)

    # seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    s = ('\n' + '%10s' * 4) % ('L1', 'L2', 'psnr', 'loss_out')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # jdict, stats, ap, ap_class = [], [], [], []
    if not compute_loss_me:
        compute_loss_me = ComputeLossME(device, feature_max, 1)
    stats_residual = StatCalculator(dist_range, bins) if track_stats else None
    mloss = torch.zeros(4, device=device)  # mean losses
    
    for batch_i, (ref1, ref2, target) in enumerate(tqdm(dataloader, desc=s, position=0, leave=True)):
        t1 = time_sync()
        ref1 = ref1.to(device, non_blocking=True)
        ref1 = ref1.half() if half else ref1.float()  # uint8 to fp16/32
        ref1 /= 255  # 0 - 255 to 0.0 - 1.0
        ref2 = ref2.to(device, non_blocking=True)
        ref2 = ref2.half() if half else ref2.float()  # uint8 to fp16/32
        ref2 /= 255  # 0 - 255 to 0.0 - 1.0
        target = target.to(device, non_blocking=True)
        target = target.half() if half else target.float()  # uint8 to fp16/32
        target /= 255  # 0 - 255 to 0.0 - 1.0

        nb, _, height, width = ref1.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        T1 = model(ref1, cut_model=1)       # first half of the model
        T2 = model(ref2, cut_model=1)       # first half of the model
        Ttrg = model(target, cut_model=1)   # first half of the model
        T1_tilde = autoencoder(T1, task='enc')
        T2_tilde = autoencoder(T2, task='enc')
        Ttrg_tilde = autoencoder(Ttrg, task='enc')
        Tpred_tilde = motion_estimator(T1_tilde, T2_tilde)
        Tpred_hat = autoencoder(None, task='dec', bottleneck=Tpred_tilde)
        Ttrg_hat = autoencoder(None, task='dec', bottleneck=Ttrg_tilde)
        out_pred, _ = model(None, cut_model=2, T=Tpred_hat)  # second half of the model
        out_trg, _ = model(None, cut_model=2, T=Ttrg_hat)  # second half of the model
        loss, loss_items = compute_loss_me(Tpred_tilde, Ttrg_tilde, out_pred, out_trg)

        mloss = (mloss * batch_i + loss_items) / (batch_i + 1)  # update mean losses

        if track_stats:
            residual = (Ttrg_tilde - Tpred_tilde).detach().cpu().numpy()
            stats_residual.update_stats(residual)
        
        dt[1] += time_sync() - t2

        # Compute loss
        # if compute_loss:
        #     loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Statistics per image
        # for si, pred in enumerate(out):
        #     labels = targets[targets[:, 0] == si, 1:]
        #     nl = len(labels)
        #     tcls = labels[:, 0].tolist() if nl else []  # target class
        #     path, shape = Path(paths[si]), shapes[si][0]
        #     seen += 1

        #     if len(pred) == 0:
        #         if nl:
        #             stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        #         continue

        #     # Predictions
        #     if single_cls:
        #         pred[:, 5] = 0
        #     predn = pred.clone()
        #     scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

        #     # Evaluate
        #     if nl:
        #         tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
        #         scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
        #         labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
        #         correct = process_batch(predn, labelsn, iouv)
        #         if plots:
        #             confusion_matrix.process_batch(predn, labelsn)
        #     else:
        #         correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        #     stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        #     # Save/log
        #     if save_txt:
        #         save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
        #     if save_json:
        #         save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
        #     callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # # # Output the statistics of the residuals
    if track_stats:
        stats_residual.output_stats(save_dir)
    # # # Compute statistics
    # # stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # # if len(stats) and stats[0].any():
    # #     p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    # #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    # #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # #     nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    # # else:
    # #     nt = torch.zeros(1)

    # # # Print results
    pf = '%10.4g' * 4  # print format
    LOGGER.info(pf % (mloss[0], mloss[1], mloss[2], mloss[3]))
    # # with open(save_dir / 'result.txt', 'a') as f:
    # #     form = ' \n\nClass = ' + '%s' + '\n' + 'Images = ' + '%i' + '\n' + 'Labels = ' + '%i' + '\n' + 'P =' + '%.3g' + '\n' + 'R =' + '%.3g' + '\n' + 'mAP@.5 =' + '%.3g' + '\n' + 'mAP@.5:.95 =' + '%.3g' + '\n\n'
    # #     f.write(form % ('all', seen, nt.sum(), mp, mr, map50*100, map*100))
    # #     if(add_noise):
    # #         f.write('Losses:\n')
    # #         f.write(' L1 =\t %.4f\n L2 =\t %.4f\n SATD =\t %.4f\n' % (sum(L1)/len(L1), sum(L2)/len(L2), sum(SATD)/len(SATD)))

    # # # Print results per class
    # # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    # #     for i, c in enumerate(ap_class):
    # #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # # # Print speeds
    # # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # # if not training:
    # #     shape = (batch_size, 3, imgsz, imgsz)
    # #     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # # # Plots
    # # if plots:
    # #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # #     callbacks.run('on_val_end')

    # # # Save JSON
    # # if save_json and len(jdict):
    # #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    # #     anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
    # #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    # #     LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
    # #     with open(pred_json, 'w') as f:
    # #         json.dump(jdict, f)

    # #     try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    # #         check_requirements(['pycocotools'])
    # #         from pycocotools.coco import COCO
    # #         from pycocotools.cocoeval import COCOeval

    # #         anno = COCO(anno_json)  # init annotations api
    # #         pred = anno.loadRes(pred_json)  # init predictions api
    # #         eval = COCOeval(anno, pred, 'bbox')
    # #         if is_coco:
    # #             eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
    # #         eval.evaluate()
    # #         eval.accumulate()
    # #         eval.summarize()
    # #         map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    # #     except Exception as e:
    # #         LOGGER.info(f'pycocotools unable to run: {e}')

    # # # Return results
    # # model.float()  # for training
    # # if not training:
    # #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    # #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # # maps = np.zeros(nc) + map
    # # for i, c in enumerate(ap_class):
    # #     maps[c] = ap[i]
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    return mloss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val_ME_DNN', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    # Supplemental arguments
    parser.add_argument('--autoenc-chs',  type=int, nargs='*', default=[320,192,64], help='number of channels in autoencoder')
    parser.add_argument('--supp-weights', type=str, default=None, help='initial weights path for the autoencoder')
    parser.add_argument('--data-list', type=str, default=None, help='txt file containing the validation list')
    parser.add_argument('--weights-me', type=str, default=None, help='initial weights path for the autoencoder')
    parser.add_argument('--track-stats', action='store_true', help='track the statistical properties of the residuals')
    parser.add_argument('--dist-range',  type=float, nargs='*', default=[-3,3], help='the range of the distribution')
    parser.add_argument('--bins', type=int, default=1000, help='number of bins in histogram')
    parser.add_argument('--feature-max', type=float, default=20, help='The maximum range of the bottleneck features (for PSNR calculations)')
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

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
