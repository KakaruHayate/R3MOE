"""
author: wayn391@mastertones
"""

import datetime
import os
import pathlib
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


class Saver(object):
    def __init__(
            self,
            exp_name,
            initial_global_step=-1
    ):

        self.exp_name = exp_name
        self.exp_dir = pathlib.Path('experiments') / exp_name

        # cold start
        self.global_step = initial_global_step
        self.init_time = time.time()
        self.last_time = time.time()

        # makedirs
        os.makedirs(self.exp_dir, exist_ok=True)

        # path
        self.path_log_info = os.path.join(self.exp_dir, 'log_info.txt')

        # ckpt
        os.makedirs(self.exp_dir, exist_ok=True)

        # writer
        self.writer = SummaryWriter(os.path.join(self.exp_dir, 'logs'))

    def log_info(self, msg):
        """log method"""
        if isinstance(msg, dict):
            msg_list = []
            for k, v in msg.items():
                tmp_str = ''
                if isinstance(v, int):
                    tmp_str = '{}: {:,}'.format(k, v)
                else:
                    tmp_str = '{}: {}'.format(k, v)

                msg_list.append(tmp_str)
            msg_str = '\n'.join(msg_list)
        else:
            msg_str = msg

        # dsplay
        print(msg_str)

        # save
        with open(self.path_log_info, 'a') as fp:
            fp.write(msg_str + '\n')

    def log_value(self, dict):
        for k, v in dict.items():
            self.writer.add_scalar(k, v, self.global_step)

    def log_figure(self, dict):
        for k, v in dict.items():
            self.writer.add_figure(k, v, global_step=self.global_step)

    def get_interval_time(self, update=True):
        cur_time = time.time()
        time_interval = cur_time - self.last_time
        if update:
            self.last_time = cur_time
        return time_interval

    def get_total_time(self, to_str=True):
        total_time = time.time() - self.init_time
        if to_str:
            total_time = str(datetime.timedelta(
                seconds=total_time))[:-5]
        return total_time

    def save_model(
            self,
            model,
            name='model',
            postfix=''):
        # path
        if postfix:
            postfix = '_' + postfix
        path_pt = os.path.join(
            self.exp_dir, name + postfix + '.pt')

        # check
        print(' [*] model checkpoint saved: {}'.format(path_pt))

        # save
        torch.save(model.state_dict(), path_pt)

    def global_step_increment(self):
        self.global_step += 1
