# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict

ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
import os

def backup_code(source_dir, log_dir):
    """
    将指定目录下的代码文件复制到log目录中，但不复制logs目录下的内容
    :param source_dir: 要备份的源代码目录
    :param log_dir: 备份文件存储的log目录
    """
    # 设置备份路径
    code_backup_dir = os.path.join(log_dir, 'code_backup')

    # 如果备份目录不存在，则创建该目录
    if not os.path.exists(code_backup_dir):
        os.makedirs(code_backup_dir)

    # 计算源目录和备份目录的公共前缀
    common_prefix = os.path.commonprefix([source_dir, code_backup_dir])

    # 遍历指定目录下的所有文件和文件夹
    for root, dirs, files in os.walk(source_dir):
        # 如果当前目录为logs文件夹，则跳过该目录及其子目录
        if os.path.basename(root) == 'logs':
            dirs[:] = []
            continue
        # 否则，复制该目录下的所有文件和文件夹到备份目录中
        for file in files:
            source_file = os.path.join(root, file)
            backup_file = os.path.join(code_backup_dir, os.path.relpath(source_file, common_prefix))
            shutil.copy2(source_file, backup_file)
        for dir in dirs:
            source_dir = os.path.join(root, dir)
            backup_dir = os.path.join(code_backup_dir, os.path.relpath(source_dir, common_prefix))
            os.makedirs(backup_dir, exist_ok=True)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
        backup_code(source_dir='./',log_dir=path)

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)