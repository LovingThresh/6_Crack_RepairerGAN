# -*- coding: utf-8 -*-
# @Time    : 2023/3/27 8:56
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Crack500_Processes.py
# @Software: PyCharm
import os
import cv2
import numpy as np

# 将crack_500数据集中的图片进行处理，将他们的尺寸统一为448*448
# crack_500的文件分别存在img_dir与ann_dir, 其中各存在train, val, test三个文件夹
# img_dir中的文件为jpg格式，图像通道为3，ann_dir中的文件为png格式， 图像通道为1
# 对ann_dir中的文件进行处理，使其值为二值， 阈值设为100，分别为0和255
# 将处理后的图片存储在out_dir中
# out_crack_dir = '../2_datasets/crack_datasets/500_crack/'


def resize_image(crack_dir, size, out_dir):
    img_dir = crack_dir + 'img_dir/'
    ann_dir = crack_dir + 'ann_dir/'
    # 创建out_dir中的文件夹
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in ['img_dir', 'ann_dir']:
        for j in ['train', 'val', 'test']:
            if not os.path.exists(out_dir + i + '/' + j):
                os.makedirs(out_dir + i + '/' + j)
    for i in ['train', 'val', 'test']:
        for j in os.listdir(img_dir + i):
            img_path = img_dir + i + '/' + str(j)
            ann_path = ann_dir + i + '/' + str(j)[0:-4] + '.png'
            img = cv2.imread(img_path)
            ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            img, ann = cv2.resize(img, size), cv2.resize(ann, size)
            ann[ann < 100] = 0
            ann[ann >= 100] = 255
            # 判断图像与掩膜同时为numpy的uint8类型，抛出警告
            if not (isinstance(img, np.ndarray) and isinstance(ann, np.ndarray)):
                print('Image or Mask is not a numpy array.')
            if not (img.dtype == np.uint8 and ann.dtype == np.uint8):
                print('Image or Mask is not uint8.')
            # 判断图像与掩膜的尺寸是否相同，抛出警告
            if not (img.shape[:2] == ann.shape[:2]):
                print('Image and Mask shape mismatch: {} != {}'.format(img.shape, ann.shape))
            cv2.imwrite(out_dir + 'img_dir/' + i + '/' + str(j), img)
            cv2.imwrite(out_dir + 'ann_dir/' + i + '/' + str(j)[:-4] + '.png', ann)
    print('Resize Done!')
    return


crack_path = '../2_datasets/crack_datasets/Crack500/'
out_path = '../2_datasets/crack_datasets/500_crack/'
resize_image(crack_path, (448, 448), out_path)
