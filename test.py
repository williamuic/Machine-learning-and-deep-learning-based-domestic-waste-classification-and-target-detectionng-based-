from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.layout_dataset import LayoutData
from detectors.detector_factory import detector_factory


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = LayoutData
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_path = os.path.join(dataset.img_dir, '{}.jpg'.format(img_id))
        # 得到图像路径

        ret = detector.run(img_path, img_id)
        # 得到检测框

        result = ret['results']
        for i in range(len(result)):
            c = result[i][-1]
            result[i].append(dataset.class_name[c])
        results[img_id] = ret['results']
    print(results)
    # results为最终得到的预测框

if __name__ == '__main__':
    opt = opts().parse()
    test(opt)



