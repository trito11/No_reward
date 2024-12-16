import os
import shutil
import argparse
import logging
import time
import getpass
import numpy as np
from termcolor import colored
from beautifultable import BeautifulTable
from torch.utils.tensorboard import SummaryWriter  # PyTorch

def str2bool(value):
    value = str(value)
    if isinstance(value, bool):
       return value
    if value.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif value.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got ' + str(value.lower()))

def make_dir(dir_name, clear=True):
    if os.path.exists(dir_name):
        if clear:
            try: shutil.rmtree(dir_name)
            except: pass
            try: os.makedirs(dir_name)
            except: pass
    else:
        try: os.makedirs(dir_name)
        except: pass

def dir_ls(dir_path):
    dir_list = os.listdir(dir_path)
    dir_list.sort()
    return dir_list

def system_pause():
    getpass.getpass("Press Enter to Continue")

def get_arg_parser():
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def remove_color(key):
    for i in range(len(key)):
        if key[i] == '@':
            return key[:i]
    return key

def load_npz_info(file_path):
    return np.load(file_path)['info'][()]

class Logger:
    def __init__(self, name):
        make_dir('log', clear=False)
        make_dir('log/text', clear=False)
        if name is None:
            self.name = time.strftime('%Y-%m-%d-%H:%M:%S')
        else:
            self.name = name + time.strftime('-(%Y-%m-%d-%H:%M:%S)')

        log_file = 'log/text/' + self.name + '.log'
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        self.tabular_reset()

    def debug(self, *args):
        self.logger.debug(*args)

    def info(self, *args):
        self.logger.info(*args)

    def warning(self, *args):
        self.logger.warning(*args)

    def error(self, *args):
        self.logger.error(*args)

    def critical(self, *args):
        self.logger.critical(*args)

    def log_time(self, log_tag=''):
        log_info = time.strftime('%Y-%m-%d %H:%M:%S')
        if log_tag != '':
            log_info += ' ' + log_tag
        self.info(log_info)

    def tabular_reset(self):
        self.keys = []
        self.colors = []
        self.values = {}
        self.counts = {}
        self.summary = []

    def tabular_clear(self):
        for key in self.keys:
            self.counts[key] = 0

    def summary_init(self):
        make_dir('log/board', clear=False)
        self.summary_writer = SummaryWriter(log_dir='log/board/' + self.name)

    def summary_clear(self):
        if hasattr(self, 'summary_writer'):
            self.summary_writer.flush()

    def summary_show(self, steps):
        if hasattr(self, 'summary_writer'):
            self.summary_writer.flush()

    def check_color(self, key):
        for i in range(len(key)):
            if key[i] == '@':
                return key[:i], key[i+1:]
        return key, None
    def summary_setup(self):
        """Thiết lập cho TensorBoard summary."""
        # Chắc chắn rằng summary_writer đã được khởi tạo
        if hasattr(self, 'summary_writer'):
            # Thực hiện các thiết lập cần thiết cho summary_writer
            self.summary_writer.setup()
        else:
            raise AttributeError("summary_writer chưa được khởi tạo. Hãy gọi summary_init trước.")

    def add_item(self, key, summary_type='none'):
        assert not (key in self.keys)
        key, color = self.check_color(key)
        self.counts[key] = 0
        self.keys.append(key)
        self.colors.append(color)
        if summary_type != 'none':
            assert hasattr(self, 'summary_writer')
            self.summary.append(key)

    def add_record(self, key, value, count=1):
        key, _ = self.check_color(key)
        if type(value) == np.ndarray:
            count *= np.prod(value.shape)
            value = np.mean(value)  # convert to scalar
        if self.counts[key] > 0:
            self.values[key] += value * count
            self.counts[key] += count
        else:
            self.values[key] = value * count
            self.counts[key] = count
        if key in self.summary:
            self.summary_writer.add_scalar(key, value, count)

    def add_dict(self, info, prefix='', count=1):
        for key, value in info.items():
            self.add_record(prefix + key, value, count)

    def tabular_show(self, log_tag=''):
        table = BeautifulTable()
        table_c = BeautifulTable()
        for key, color in zip(self.keys, self.colors):
            if self.counts[key] == 0:
                value = ''
            elif self.counts[key] == 1:
                value = self.values[key]
            else:
                value = self.values[key] / self.counts[key]
            key_c = key if color is None else colored(key, color, attrs=['bold'])
            table.append_row([key, value])
            table_c.append_row([key_c, value])

        def customize(table):
            table.set_style(BeautifulTable.STYLE_NONE)
            table.left_border_char = '|'
            table.right_border_char = '|'
            table.column_separator_char = '|'
            table.top_border_char = '-'
            table.bottom_border_char = '-'
            table.intersect_top_left = '+'
            table.intersect_top_mid = '+'
            table.intersect_top_right = '+'
            table.intersect_bottom_left = '+'
            table.intersect_bottom_mid = '+'
            table.intersect_bottom_right = '+'
            table.column_alignments[0] = BeautifulTable.ALIGN_LEFT
            table.column_alignments[1] = BeautifulTable.ALIGN_LEFT

        customize(table)
        customize(table_c)
        self.log_time(log_tag)
        self.debug(table)
        print(table_c)

    def save_npz(self, info, info_name, folder, subfolder=''):
        make_dir('log/' + folder, clear=False)
        make_dir('log/' + folder + '/' + self.name, clear=False)
        if subfolder != '':
            make_dir('log/' + folder + '/' + self.name + '/' + subfolder, clear=False)
            save_path = 'log/' + folder + '/' + self.name + '/' + subfolder
        else:
            save_path = 'log/' + folder + '/' + self.name
        np.savez(save_path + '/' + info_name + '.npz', info=info)

def get_logger(name=None):
    return Logger(name)
