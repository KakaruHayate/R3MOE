import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import schedulefree
import tqdm
import yaml

import logger.utils
from lib import dataset, nets
from logger import utils
from logger.saver import Saver


def calc_r_squared(y_true, y_pred):
    # R-squared: r^2 = 1 - (SSE / SST)
    ss_res = torch.sum((y_pred - y_true) ** 2)
    mean_y_true = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean_y_true) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    return r2


class JSDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)

        m = 0.5 * (p + q)

        kl_p_m = F.kl_div(m.log(), p, reduction=self.reduction)
        kl_q_m = F.kl_div(m.log(), q, reduction=self.reduction)

        js_divergence = 0.5 * (kl_p_m + kl_q_m)

        return js_divergence


class KLDivLoss(torch.nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, p, q):
        kl_p = F.kl_div(
            F.log_softmax(p,dim=1), 
            F.softmax(q,dim=1), 
            reduction=self.reduction
            )
        kl_q = F.kl_div(
            F.log_softmax(q,dim=1), 
            F.softmax(p,dim=1), 
            reduction=self.reduction
            )

        kl_divergence = (kl_p + kl_q) * 0.5

        return kl_divergence


def train_epoch(dataloader, model, device, optimizer, saver, epoch):
    model.train()
    optimizer.train()
    sum_loss = 0
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(reduction='mean') # JS散度似乎也不错
    # r_drop_loss_fn = KLDivLoss(reduction='batchmean')
    # r_drop_loss_fn = JSDivLoss(reduction='batchmean')
    # r_drop_loss_fn = nn.MSELoss()
    # r_drop_loss_fn = nn.HuberLoss(reduction='mean')

    for itr, (X_gt, y_gt) in enumerate(dataloader):
        # X: [B, T, in_dims], y: [B, T]
        saver.global_step_increment()
        X_gt = X_gt.to(device)
        y_gt = y_gt.to(device)
        l_pred1 = model(X_gt)
        l_pred2 = model(X_gt.detach())
        l_gt = model.normalize(y_gt)
        loss = (criterion(l_pred1, l_gt) + criterion(l_pred2, l_gt)) * 0.5
        r_drop_loss = criterion(l_pred1, l_pred2)
        total_loss = loss + 0.5 * r_drop_loss
        current_lr = optimizer.param_groups[0]['lr']
        if saver.global_step % 10 == 0:
            saver.log_info(
                'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.6f} | r_drop_loss: {:.6f} | time: {} | step: {}'
                .format(
                    epoch,
                    itr,
                    len(dataloader),
                    saver.exp_name,
                    10 / saver.get_interval_time(),
                    current_lr,
                    loss.item(),
                    r_drop_loss.item(),
                    saver.get_total_time(),
                    saver.global_step
                )
            )
        if saver.global_step % 100 == 0:
            saver.log_value({
                'train/epoch': epoch,
                'train/loss': loss.item(),
                'train/r_drop_loss': r_drop_loss.item(),
                'train/total_loss': total_loss.item(),
                'train/lr': current_lr
            })

        total_loss.backward()
        optimizer.step()
        model.zero_grad()

        sum_loss += total_loss.item() * len(X_gt)

    return sum_loss / len(dataloader.dataset)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', '-N', type=str, default="model_test")
    p.add_argument('--dataset', '-d', required=True)
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
        'vmin': args.vmin,
        'vmax': args.vmax,
        'hidden_dims': args.hidden_dims,
        'n_layers': args.n_layers,
        'conv_dropout': args.conv_dropout
    }
    model = nets.BiLSTMCurveEstimator(**model_args)
    if args.pretrained_model is not None:
        print("loading pretrained model: " + args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

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
        _ = train_epoch(train_dataloader, model, device, optimizer, saver, epoch)
        val_loss = validate_epoch(
            val_dataloader, model, device, optimizer, saver,
            draw=(epoch + 1) % args.plot_epoch_interval == 0
        )

        if (epoch + 1) % args.save_epoch_interval == 0:
            saver.save_model(model, postfix=str(epoch))


if __name__ == '__main__':
    main()
