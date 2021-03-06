# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
from models.google_video import IntraEncoder

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val_end_to_end  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.supplemental import AutoEncoder
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.datasets import create_dataloader
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer, print_args_to_file)
from utils.downloads import attempt_download
from utils.loss import ComputeLoss, RateDistortionLoss, compute_aux_loss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness_e2e
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

print(LOCAL_RANK, RANK, WORLD_SIZE)


def configure_optimizers(net: nn.Module, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer and the yolo optimizer.
    Return three optimizers"""
    ### intra_encoder ###
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


def train_intra(hyp,  # path/to/hyp.yaml or hyp dictionary
                opt,
                device,
                callbacks
                ):
    save_dir, epochs, batch_size, weights, single_cls, data, resume, noval, nosave, workers, intra_weights= \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.intra_weights

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'yolo_last.pt', w / 'yolo_best.pt'
    net_last, net_best = w / 'intra_last.pt', w / 'intra_best.pt'

    print_args_to_file(opt, save_dir / 'result.txt')
    opt.evolve = False

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = True  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    assert weights.endswith('.pt')
    with torch_distributed_zero_first(LOCAL_RANK):
        weights = attempt_download(weights)  # download if not found locally
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (hyp.get('anchors')) and not resume else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

    net = IntraEncoder(*opt.autoenc_chs).to(device)
    net_pretrained = False
    if intra_weights is not None:
        net_pretrained = intra_weights.endswith('.pt')
        if net_pretrained:
            enc_ckpt = torch.load(intra_weights, map_location=device)
            net.load_state_dict(enc_ckpt['state_dict'])
            print('pretrained intra-encoder')
            del enc_ckpt

    encoder_optimizer, aux_optimizer = configure_optimizers(net, opt)
    
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    cut_stride = 8
    encoder_stride = 64
    gs = max(gs, cut_stride*encoder_stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs)  # verify imgsz is gs-multiple

    # Scheduler
    encoder_scheduler = lr_scheduler.ReduceLROnPlateau(encoder_optimizer, "min", patience=4)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, -100.0
    if resume:
        # Optimizer
        best_fitness = ckpt['best_fitness']
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']
        # Epochs
        start_epoch = ckpt['epoch'] + 1
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs
        del ckpt, csd

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=False, rank=LOCAL_RANK,
                                              workers=workers, image_weights=False, quad=False,
                                              prefix=colorstr('train: '))
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.0,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
            net.half().float()

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        net = DDP(net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    maps = np.zeros(nc)  # mAP per class
    # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    criterion = RateDistortionLoss(lmbda=opt.lmbda, return_details=True)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.eval()
        net.train()

        mloss = torch.zeros(9, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 13) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'obj_tot', 'tot', 'fid', 'bpp', 'enc', 'aux', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        aux_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Forward
            T = model(imgs, cut_model=1)  # first half of the model
            # T = (T + 1) / 20
            out_net = net(T)
            T_hat = out_net['x_hat'][0]
            # pred = model(None, cut_model=2, T=T_hat*20-1)[1]  # second half of the model
            pred = model(None, cut_model=2, T=T_hat)[1]  # second half of the model

            out_criterion = criterion(out_net, [T])
            loss_o, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            
            loss = (1-opt.w_obj_det)*out_criterion["loss"] + opt.w_obj_det*loss_o
            aux_loss = compute_aux_loss(net.aux_loss(), backward=True)
            loss_items = torch.cat((loss_items, loss_o, loss, out_criterion["distortion"].unsqueeze(0), out_criterion["bpp_loss"].unsqueeze(0), out_criterion["loss"].unsqueeze(0), aux_loss.unsqueeze(0)))
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)

            # Optimize
            aux_optimizer.step()
            encoder_optimizer.step  # optimizer.step
            if ema:
                ema.update(model)

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 11) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)
            # end batch ------------------------------------------------------------------------------------------------

        if RANK in [-1, 0]:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val_end_to_end.run(data_dict,
                                                      batch_size=batch_size // WORLD_SIZE * 2,
                                                      imgsz=imgsz,
                                                      model=ema.ema,
                                                      single_cls=single_cls,
                                                      dataloader=val_loader,
                                                      save_dir=save_dir,
                                                      plots=False,
                                                      callbacks=callbacks,
                                                      compute_loss=compute_loss,
                                                      intra_encoder=deepcopy(net).half(),
                                                      criterion=criterion)

            # results : mp, mr, map50, map, box, obj, cls, loss_o, mse_loss, bpp_loss, enc_loss, aux_loss
            loss_val = (1-opt.w_obj_det)*results[10] + opt.w_obj_det*results[7]
            encoder_scheduler.step(loss_val)
            results += (loss_val,)

            # Update best mAP
            fi = fitness_e2e(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + [encoder_optimizer.param_groups[0]['lr']]
            callbacks.run('on_fit_epoch_end_e2e', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}
                net_ckpt = {"epoch": epoch,
                            "state_dict": net.state_dict(),
                            "loss": loss,
                            "optimizer": encoder_optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": encoder_scheduler.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                torch.save(net_ckpt, net_last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    torch.save(net_ckpt, net_best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt, net_ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    if RANK in [-1, 0]:
        del model, net
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:

                    best_intra_encoder = IntraEncoder(*opt.autoenc_chs).to(device)
                    net_ckpt = torch.load(net_best)
                    best_intra_encoder.load_state_dict(net_ckpt['state_dict'])

                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val_end_to_end.run(data_dict,
                                                       batch_size=batch_size // WORLD_SIZE * 2,
                                                       imgsz=imgsz,
                                                       model=attempt_load(f, device).half(),
                                                       iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                                       single_cls=single_cls,
                                                       dataloader=val_loader,
                                                       save_dir=save_dir,
                                                       save_json=is_coco,
                                                       verbose=True,
                                                       plots=True,
                                                       callbacks=callbacks,
                                                       compute_loss=compute_loss,      # val best model with plots
                                                       intra_encoder=best_intra_encoder.half(),
                                                       criterion=criterion)

                    # results : mp, mr, map50, map, box, obj, cls, loss_o, mse_loss, bpp_loss, enc_loss, aux_loss
                    loss_val = (1-opt.w_obj_det)*results[10] + opt.w_obj_det*results[7]
                    results += (loss_val,)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end_e2e', list(mloss) + list(results) + [encoder_optimizer.param_groups[0]['lr']], epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5x6.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train_end2end', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    # Supplemental arguments
    parser.add_argument('--autoenc-chs',  type=int, nargs='*', default=[320, 256, 192], help='number of channels in autoencoder')
    parser.add_argument('--intra-weights', type=str, default=None, help='initial weights path for the intra_encoder')
    parser.add_argument('--inter-weights', type=str, default=None, help='initial weights path for the inter_encoder')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help="Learning rate (default: %(default)s)")
    parser.add_argument('--aux-learning-rate', type=float, default=1e-3, help="Auxiliary loss learning rate (default: %(default)s)")
    parser.add_argument('--train-list', type=str, default=None, help='txt file containing the training list')
    parser.add_argument('--val-list', type=str, default=None, help='txt file containing the validation list')
    parser.add_argument('--w-obj-det', type=float, default=1, help='The weight of the object detection loss in the total loss')
    # parser.add_argument('--deform-G', type=int, default=1, help='number of groups in deformable convolution layers')
    parser.add_argument('--frame-type', type=str, default='intra', help='type of the frame to be trained (intra/inter)')
    parser.add_argument('--lambda', dest="lmbda", type=float, default=1e-2, help="Bit-rate distortion parameter (default: %(default)s)")
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume = ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.weights), '--weights must be specified'
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if opt.frame_type == 'intra':
        train_intra(opt.hyp, opt, device, callbacks)
    # else:
    #     train_inter(opt.hyp, opt, device, callbacks)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()

def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
