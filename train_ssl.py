import argparse
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import schedulefree
import tqdm
import yaml

import logger.utils
from lib import dataset, nets, ramps
from logger import utils
from logger.saver import Saver


def calc_r_squared(y_true, y_pred):
    # R-squared: r^2 = 1 - (SSE / SST)
    ss_res = torch.sum((y_pred - y_true) ** 2)
    mean_y_true = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean_y_true) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    return r2


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 10 * ramps.sigmoid_rampup(epoch, 10)


def train_epoch(dataloader, model, device, optimizer, saver, epoch, ema_model, dataloader_unlabel):
    model.train()
    optimizer.train()
    sum_loss = 0
    total_samples = 0
    criterion = nn.HuberLoss(reduction='mean') 
    # 目前看来huberloss效果是最好的，各种场合，但从数学表示看工作中应该和MSE效果差不多？

    iter_labeled = itertools.cycle(dataloader)
    iter_unlabeled = iter(dataloader_unlabel) if len(dataloader_unlabel) > len(dataloader) else itertools.cycle(dataloader_unlabel)
    max_iters = max(len(dataloader), len(dataloader_unlabel))

    for itr in range(max_iters):
        saver.global_step_increment()

        consistency_weight = get_current_consistency_weight(epoch)
        
        X_gt, y_gt = next(iter_labeled)
        X_unlabel, X_unlabel1, X_unlabel2 = next(iter_unlabeled)
        # X_unlabel: teacher输入的无扰动样本
        # X_unlabel1, X_unlabel2: student输入的有扰动样本

        X_gt = X_gt.to(device)
        y_gt = y_gt.to(device)
        X_unlabel = X_unlabel.to(device)
        X_unlabel1 = X_unlabel1.to(device)
        X_unlabel2 = X_unlabel2.to(device)

        # 监督学习部分
        l_pred1 = model(X_gt)
        l_pred2 = model(X_gt.detach())
        l_gt = model.normalize(y_gt)
        loss = (criterion(l_pred1, l_gt) + criterion(l_pred2, l_gt)) * 0.5
        r_drop_loss = criterion(l_pred1, l_pred2)

        # 一致性正则化
        with torch.no_grad():
            ema_model.eval()
            fakelabel_pred = ema_model.module(X_unlabel)
        unlabeled_pred1 = model(X_unlabel1)
        unlabeled_pred2 = model(X_unlabel2)
        consistency_loss = (criterion(unlabeled_pred1, fakelabel_pred) + criterion(unlabeled_pred2, fakelabel_pred)) * 0.5

        total_loss = (loss + 0.5 * r_drop_loss) + consistency_loss * consistency_weight

        current_lr = optimizer.param_groups[0]['lr']
        if saver.global_step % 10 == 0:
            saver.log_info(
                'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.6f} | r_drop_loss: {:.6f} | consistency_loss: {:.6f} | time: {} | step: {}'
                .format(
                    epoch,
                    itr,
                    max_iters,
                    saver.exp_name,
                    10 / saver.get_interval_time(),
                    current_lr,
                    loss.item(),
                    r_drop_loss.item(),
                    consistency_loss.item(),
                    saver.get_total_time(),
                    saver.global_step
                )
            )
        if saver.global_step % 100 == 0:
            saver.log_value({
                'train/epoch': epoch,
                'train/loss': loss.item(),
                'train/r_drop_loss': r_drop_loss.item(),
                'train/consistency_loss': consistency_loss.item(),
                'train/total_loss': total_loss.item(),
                'train/consistency_weight': consistency_weight,
                'train/lr': current_lr
            })

        total_loss.backward()
        optimizer.step()
        ema_model.update_parameters(model)
        model.zero_grad()

        batch_samples = len(X_gt) + len(X_unlabel)
        sum_loss += total_loss.item() * batch_samples
        total_samples += batch_samples

    return sum_loss / total_samples if total_samples > 0 else 0


def validate_epoch(dataloader, model, device, optimizer, saver, draw=False):
    model.eval()
    optimizer.eval()

    sum_loss = 0
    sum_mae = 0
    gt_cache = [] # 在整个验证集取r^2，所以把gt和pred先cache在concat到一起
    pred_cache = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for idx, (X_gt, y_gt) in enumerate(
                tqdm.tqdm(dataloader, total=len(dataloader), desc='validation', leave=False)
        ):
            # X: [B, T, in_dims], y: [B, T]
            X_gt = X_gt.to(device)
            y_gt = y_gt.to(device)
            l_pred = model(X_gt)
            l_gt = model.normalize(y_gt)
            loss = criterion(l_pred, l_gt)
            sum_loss += loss.item() * len(X_gt)
            y_pred = model.denormalize(l_pred)
            gt_cache.append(y_gt)
            pred_cache.append(y_pred)
            sum_mae += torch.nn.functional.l1_loss(y_pred, y_gt).detach().cpu().numpy()
            if not draw:
                continue
            spec_draw = X_gt[0].cpu().numpy()
            curve_gt_draw = y_gt[0].cpu().numpy()
            curve_pred_draw = y_pred[0].cpu().numpy()
            if spec_draw.shape[0] > 1024:
                spec_draw = spec_draw[:1024]
                curve_gt_draw = curve_gt_draw[:1024]
                curve_pred_draw = curve_pred_draw[:1024]
            saver.log_figure({
                f'curve_{idx}': logger.utils.draw_plot(
                    spec=spec_draw,
                    curve_gt=curve_gt_draw,
                    curve_pred=curve_pred_draw
                )
            })

    mean_loss = sum_loss / len(dataloader.dataset)
    r_squared = calc_r_squared(torch.cat(gt_cache, dim=1), torch.cat(pred_cache, dim=1))
    mean_mae = sum_mae / len(dataloader.dataset)
    saver.log_info(' --- <validation> --- loss: {:.6f} MAE: {:.6f} R_squared: {:.6f}'.format(mean_loss, mean_mae, r_squared))
    saver.log_value({
        'validation/loss': mean_loss,
        'validation/mae': mean_mae,
        'validation/r_squared': r_squared
    })
    return mean_loss


def validate_epoch2(dataloader, model, device, optimizer, saver, draw=False):
    model.eval()
    optimizer.eval()

    sum_loss = 0
    sum_mae = 0
    gt_cache = [] # 在整个验证集取r^2，所以把gt和pred先cache在concat到一起
    pred_cache = []
    criterion = nn.MSELoss()

    with torch.no_grad():
        for idx, (X_gt, y_gt) in enumerate(
                tqdm.tqdm(dataloader, total=len(dataloader), desc='validation_unseen', leave=False)
        ):
            # X: [B, T, in_dims], y: [B, T]
            X_gt = X_gt.to(device)
            y_gt = y_gt.to(device)
            l_pred = model(X_gt)
            l_gt = model.normalize(y_gt)
            loss = criterion(l_pred, l_gt)
            sum_loss += loss.item() * len(X_gt)
            y_pred = model.denormalize(l_pred)
            gt_cache.append(y_gt)
            pred_cache.append(y_pred)
            sum_mae += torch.nn.functional.l1_loss(y_pred, y_gt).detach().cpu().numpy()
            if not draw:
                continue
            spec_draw = X_gt[0].cpu().numpy()
            curve_gt_draw = y_gt[0].cpu().numpy()
            curve_pred_draw = y_pred[0].cpu().numpy()
            if spec_draw.shape[0] > 1024:
                spec_draw = spec_draw[:1024]
                curve_gt_draw = curve_gt_draw[:1024]
                curve_pred_draw = curve_pred_draw[:1024]
            saver.log_figure({
                f'curve_unseen_{idx}': logger.utils.draw_plot(
                    spec=spec_draw,
                    curve_gt=curve_gt_draw,
                    curve_pred=curve_pred_draw
                )
            })

    mean_loss = sum_loss / len(dataloader.dataset)
    r_squared = calc_r_squared(torch.cat(gt_cache, dim=1), torch.cat(pred_cache, dim=1))
    mean_mae = sum_mae / len(dataloader.dataset)
    saver.log_info(' --- <validation> --- loss_unseen: {:.6f} MAE_unseen: {:.6f} R_squared_unseen: {:.6f}'.format(mean_loss, mean_mae, r_squared))
    saver.log_value({
        'validation/loss_unseen': mean_loss,
        'validation/mae_unseen': mean_mae,
        'validation/r_squared_unseen': r_squared
    })
    return mean_loss


def draw_unlabel(dataloader, model, device, optimizer, saver, draw=True):
    model.eval()
    optimizer.eval()
    mean_loss = 0
    with torch.no_grad():
        for idx, X_gt in enumerate(
                tqdm.tqdm(dataloader, total=len(dataloader), desc='plot', leave=False)
        ):
            # X: [B, T, in_dims], y: [B, T]
            if not draw:
                continue
            X_gt = X_gt.to(device)
            l_pred = model(X_gt)
            y_pred = model.denormalize(l_pred)
            spec_draw = X_gt[0].cpu().numpy()
            curve_pred_draw = y_pred[0].cpu().numpy()
            if spec_draw.shape[0] > 1024:
                spec_draw = spec_draw[:1024]
                curve_pred_draw = curve_pred_draw[:1024]
            saver.log_figure({
                f'curve_unlabel_{idx}': logger.utils.draw_plot(
                    spec=spec_draw,
                    curve_gt=curve_pred_draw, # 兼容性复用
                    curve_pred=curve_pred_draw
                )
            })
        saver.log_info('draw unlabel done')
    return mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', '-N', type=str, default="model_test")
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--unlabel_dataset', '-ud', required=True)
    p.add_argument('--vmin', type=float, default=0.)
    p.add_argument('--vmax', type=float, default=1.)
    p.add_argument('--batchsize', '-B', type=int, default=16)
    p.add_argument('--cropsize', '-C', type=int, default=128)
    p.add_argument('--learning_rate', '-l', type=float, default=0.0005)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=3047)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=200)
    p.add_argument('--hidden_dims', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--conv_dropout', type=float, default=0.2)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--plot_epoch_interval', type=int, default=1)
    p.add_argument('--save_epoch_interval', type=int, default=1)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.conv_dropout==0:
        raise ValueError("You must enable dropout for building positive samples!!!")

    # unlabel数据集
    train_dataset_unlabel = dataset.UnlabelTrainingDataset(
        args.unlabel_dataset,
        crop_size=args.cropsize,
        volume_aug_rate=0.5
    )
    train_dataloader_unlabel = torch.utils.data.DataLoader(
        dataset=train_dataset_unlabel,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )
    # label数据集
    train_dataset = dataset.CurveTrainingDataset(
        args.dataset,
        crop_size=args.cropsize,
        volume_aug_rate=0.5
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )
    # seen测试集
    val_dataset = dataset.CurveValidationDataset(args.dataset)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # unseen测试集
    val_dataset2 = dataset.CurveValidationDataset2(args.dataset)
    val_dataloader2 = torch.utils.data.DataLoader(
        dataset=val_dataset2,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # unlabel测试集
    val_dataset_unlabel = dataset.CurveValidationDatasetUnlabel(args.unlabel_dataset)
    val_dataloader_unlabel = torch.utils.data.DataLoader(
        dataset=val_dataset_unlabel,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device('cpu')
    model_args = {
        'in_dims': train_dataset.metadata['mel_bins'],
        'vmin': args.vmin,
        'vmax': args.vmax,
        'hidden_dims': args.hidden_dims,
        'n_layers': args.n_layers,
        'conv_dropout': args.conv_dropout
    }

    # student model
    model = nets.BiLSTMCurveEstimator(**model_args)

    if args.pretrained_model is not None:
        print("loading pretrained model: " + args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    # teacher model
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )

    optimizer = schedulefree.AdamWScheduleFree(
        filter(lambda parameter: parameter.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )

    saver = Saver(args.exp_name)
    with open(saver.exp_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump({
            "dataset_args": dict(train_dataset.metadata),
            "model_args": model_args
        }, f)

    params_count = utils.get_network_paras_amount({'model': model})
    print(model)
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    for epoch in range(args.epoch):
        # 训练循环
        _ = train_epoch(
            train_dataloader, model, device, optimizer, saver, epoch, ema_model, train_dataloader_unlabel
        )
        # seen测试集
        val_loss = validate_epoch(
            val_dataloader, ema_model.module, device, optimizer, saver,
            draw=(epoch + 1) % args.plot_epoch_interval == 0
        )
        # unseen测试集
        val_loss2 = validate_epoch2(
            val_dataloader2, ema_model.module, device, optimizer, saver,
            draw=(epoch + 1) % args.plot_epoch_interval == 0
        )
        # unlabel测试集
        _ = draw_unlabel(
            val_dataloader_unlabel, ema_model.module, device, optimizer, saver,
            draw=(epoch + 1) % args.plot_epoch_interval == 0
        )

        if (epoch + 1) % args.save_epoch_interval == 0:
            saver.save_model(model, postfix=str(epoch))
            saver.save_model(ema_model.module, name='ema_model', postfix=str(epoch))


if __name__ == '__main__':
    main()
