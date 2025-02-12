import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import yaml

import logger.utils
from lib import dataset, nets
from logger import utils
from logger.saver import Saver


def train_epoch(dataloader, model, device, optimizer, saver, epoch):
    model.train()
    sum_loss = 0
    criterion = nn.BCELoss()

    for itr, (X_gt, y_gt) in enumerate(dataloader):
        # X: [B, T, in_dims], y: [B, T, out_dims]
        saver.global_step_increment()
        X_gt = X_gt.to(device)
        y_gt = y_gt.to(device)
        l_gt = model.encode(y_gt)
        l_pred = model(X_gt)
        loss = criterion(l_pred, l_gt)

        current_lr = optimizer.param_groups[0]['lr']
        if saver.global_step % 10 == 0:
            saver.log_info(
                'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.6f} | time: {} | step: {}'
                .format(
                    epoch,
                    itr,
                    len(dataloader),
                    saver.exp_name,
                    10 / saver.get_interval_time(),
                    current_lr,
                    loss.item(),
                    saver.get_total_time(),
                    saver.global_step
                )
            )
        if saver.global_step % 100 == 0:
            saver.log_value({
                'train/epoch': epoch,
                'train/loss': loss.item(),
                'train/lr': current_lr
            })

        loss.backward()
        optimizer.step()
        model.zero_grad()

        sum_loss += loss.item() * len(X_gt)

    return sum_loss / len(dataloader.dataset)


def validate_epoch(dataloader, model, device, saver, draw=False):
    model.eval()

    sum_loss = 0
    sum_mse = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for idx, (X_gt, y_gt) in enumerate(
                tqdm.tqdm(dataloader, total=len(dataloader), desc='validation', leave=False)
        ):
            # X: [B, T, in_dims], y: [B, T, out_dims]
            X_gt = X_gt.to(device)
            y_gt = y_gt.to(device)
            l_gt = model.encode(y_gt)
            l_pred = model(X_gt)
            loss = criterion(l_pred, l_gt)
            sum_loss += loss.item() * len(X_gt)
            y_pred = model.decode(l_pred)
            sum_mse += ((y_pred - y_gt) ** 2).mean().item()
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
    mean_mse = sum_mse / len(dataloader.dataset)
    saver.log_info(' --- <validation> --- loss: {:.6f}, mse: {:.6f} '.format(mean_loss, mean_mse))
    saver.log_value({
        'validation/loss': mean_loss,
        'validation/mse': mean_mse
    })
    return mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', '-N', type=str, default="model_test")
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--vmin', type=float, default=0.)
    p.add_argument('--vmax', type=float, default=1.)
    p.add_argument('--deviation', type=float, default=0.01)
    p.add_argument('--batchsize', '-B', type=int, default=16)
    p.add_argument('--cropsize', '-C', type=int, default=128)
    p.add_argument('--learning_rate', '-l', type=float, default=0.0005)
    p.add_argument('--lr_min', type=float, default=0.00001)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=2019)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=200)
    p.add_argument('--out_dims', type=int, default=256)
    p.add_argument('--hidden_dims', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--use_fa_norm', action='store_true')
    p.add_argument('--conv_only', action='store_true')
    p.add_argument('--conv_dropout', type=float, default=0.)
    p.add_argument('--attn_dropout', type=float, default=0.)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--plot_epoch_interval', type=int, default=1)
    p.add_argument('--save_epoch_interval', type=int, default=1)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    val_dataset = dataset.CurveValidationDataset(args.dataset)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device('cpu')
    model_args = {
        'in_dims': train_dataset.metadata['mel_bins'],
        'out_dims': args.out_dims,
        'vmin': args.vmin,
        'vmax': args.vmax,
        'deviation': args.deviation,
        'hidden_dims': args.hidden_dims,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'use_fa_norm': args.use_fa_norm,
        'conv_only': args.conv_only,
        'conv_dropout': args.conv_dropout,
        'attn_dropout': args.attn_dropout
    }
    model = nets.CFNaiveCurveEstimator(**model_args)
    if args.pretrained_model is not None:
        print("loading pretrained model: " + args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True
    )

    saver = Saver(args.exp_name)
    with open(saver.exp_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump({
            "dataset_args": dict(train_dataset.metadata),
            "model_args": model_args
        }, f)

    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    for epoch in range(args.epoch):
        _ = train_epoch(train_dataloader, model, device, optimizer, saver, epoch)
        val_loss = validate_epoch(
            val_dataloader, model, device, saver,
            draw=(epoch + 1) % args.plot_epoch_interval == 0
        )

        scheduler.step(val_loss)
        if (epoch + 1) % args.save_epoch_interval == 0:
            saver.save_model(model, postfix=str(epoch))


if __name__ == '__main__':
    main()
