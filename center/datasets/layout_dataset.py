from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import torch
import json
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.debugger import Debugger
import math
import random

class LayoutData(data.Dataset):
    num_classes = 40
    default_resolution = [256, 256]
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(LayoutData, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'layout')
        self.img_dir = os.path.join(self.data_dir, 'images')
        txt_dir = os.path.join(self.data_dir, '{}.txt'.format(split))
        fh = open(txt_dir, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            imgs.append(line)
        self.images = imgs
        self.annot_path = os.path.join(self.data_dir,  'annotations')
        self.max_objs = 32
        self.split = split
        self.opt = opt
        self.class_name = [
            '一次性快餐盒', '水果果肉', '水果果皮', '茶叶渣', '菜叶菜根', '蛋壳', '鱼骨',
            '充电宝', '包', '化妆品瓶', '塑料玩具', '塑料碗盆', '塑料衣架', '污损塑料', '快递纸袋',
            '插头电线', '旧衣服', '易拉罐', '枕头', '毛绒玩具', '洗发水瓶', '玻璃杯', '皮鞋', '砧板',
            '烟蒂', '纸板箱', '调料瓶', '酒瓶', '金属食品罐', '锅', '食用油桶', '饮料瓶', '干电池',
            '软膏', '过期药物', '牙签', '破碎花盆及碟碗', '竹筷', '剩饭剩菜', '大骨头'
        ]
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self._valid_ids = np.arange(1, 4, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self.class_name)}
        self._data_rng = np.random.RandomState(123)
        self.num_samples = len(self.images)
        self.down_sample = 4

    def __getitem__(self, index):
        file = self.images[index]
        image_path = os.path.join(self.img_dir, '{}.jpg'.format(file))
        ann_path = os.path.join(self.annot_path, '{}.json'.format(file))
        img = cv2.imread(image_path)
        # The input image

        h, w, _ = img.shape
        if h > w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, h-w, cv2.BORDER_CONSTANT, value=0)
        elif w > h:
            img = cv2.copyMakeBorder(img, 0, w-h, 0, 0, cv2.BORDER_CONSTANT, value=0)
        h, w, _ = img.shape
        ratio = 128/h
        new_res = 128
        img = cv2.resize(img, (new_res, new_res), interpolation=cv2.INTER_AREA)
        input_h, input_w, _ = img.shape
        # Resize the image uniformly to 128*128

        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        s = max(input_h, input_w) * 1.0
        # C is the center point of the image, and S is the scale of the image, which is used for subsequent adjustment of the image

        flipped = False
        if self.split == 'train':
            sf = self.opt.scale
            cf = self.opt.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * (np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf) + 0.3)
        # get a new c and a new s

        if np.random.random() < self.opt.flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = input_w - c[0] - 1
        # Random flip


        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        # The zoom

        inp = (inp.astype(np.float32) / 255.)
        # Adjusted from 0~255 to 0~1

        if self.split == 'train':
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # Add color jitter

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        # Image standard processing

        output_h = input_h // self.down_sample
        output_w = input_w // self.down_sample
        # Get the output heatmap size

        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])


        # The following is the heatmap corresponding to the training
        center_hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        corner_hm = np.zeros((4, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        draw_gaussian = draw_umich_gaussian

        f = open(ann_path, 'r', encoding='UTF-8')
        load_dict = json.load(f)
        gt_det = []
        for k in range(len(load_dict['outputs']['object'])):
            ann = load_dict['outputs']['object'][k]
            name1, name2 = ann['name'].strip().split('/')
            cls_id = int(self.cat_ids[name2])

            bbox = []
            bbox.append((ann['bndbox']['xmin'])*ratio)
            bbox.append((ann['bndbox']['ymin'])*ratio)
            bbox.append((ann['bndbox']['xmax'])*ratio)
            bbox.append((ann['bndbox']['ymax'])*ratio)
            bbox = np.array(bbox, dtype=np.float32)

            if flipped:
                bbox[[0, 2]] = input_w - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))

            radius = int(radius * 0.85)

            left_top = np.array([bbox[0], bbox[1]], dtype=np.int32)
            right_top = np.array([bbox[2], bbox[1]], dtype=np.int32)
            left_bottom = np.array([bbox[0], bbox[3]], dtype=np.int32)
            right_bottom = np.array([bbox[2], bbox[3]], dtype=np.int32)
            draw_gaussian(corner_hm[0], left_top, radius)
            draw_gaussian(corner_hm[1], right_top, radius)
            draw_gaussian(corner_hm[2], left_bottom, radius)
            draw_gaussian(corner_hm[3], right_bottom, radius)

            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            ct_int = ct.astype(np.int32)
            draw_gaussian(center_hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg_mask[k] = 1
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'center_hm': center_hm, 'corner_hm': corner_hm, 'reg_mask': reg_mask,
               'ind': ind, 'wh': wh}


        # Below is the image output before training to check whether the groudtruth is correct
        if self.opt.draw:
            opt = self.opt
            dets_gt = np.array(gt_det).reshape(1, -1, 6)
            dets_gt[:, :, :4] *= opt.down_ratio
            for i in range(1):
                debugger = Debugger(
                    dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
                img = ret['input'].transpose(1, 2, 0)
                img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
                gt = debugger.gen_colormap(ret['center_hm'])
                gt_corner = debugger.gen_colormap(ret['corner_hm'])
                debugger.add_blend_img(img, gt, 'gt_hm_0')
                debugger.add_blend_img(img, gt_corner, 'out_gt_0')

                for k in range(len(dets_gt[i])):
                    if dets_gt[i, k, 4] > 0.1:
                        debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                            dets_gt[i, k, 4], img_id='out_gt_0')
                debugger.show_all_imgs(pause=True)
        return ret

    def __len__(self):
        return len(self.images)

    def convert_eval_format(self, all_bboxes):
        detections = [[[] for __ in range(self.num_samples)] \
                      for _ in range(self.num_classes + 1)]
        for i in range(self.num_samples):
            img_id = self.images[i]
            for j in range(1, self.num_classes + 1):
                if isinstance(all_bboxes[img_id][j], np.ndarray):
                    detections[j][i] = all_bboxes[img_id][j].tolist()
                else:
                    detections[j][i] = all_bboxes[img_id][j]
        return detections

    def save_results(self, results, save_dir):
        # json.dump(self.convert_eval_format(results),
        #           open('{}/results.json'.format(save_dir), 'w'))
        json.dump(results, open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        os.system('python tools/reval.py ' + \
                '{}/results.json'.format(save_dir))


