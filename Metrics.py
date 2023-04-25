# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 22:39
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : validation_for_GAN.py
# @Software: PyCharm

# 这个的代码的作用就是计算IoU值
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU


def M_Precision(y_true, y_pred):
    """精确率"""
    # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))
    y_pred = y_pred[:, :, :, 0:1]
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    # true positives
    tp = K.sum(K.round(K.round(K.clip(y_pred, 0, 1))
                       * K.round(K.clip(y_true_max, 0, 1))))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + 1e-8)
    return precision


def M_Recall(y_true, y_pred):
    """召回率"""
    # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))
    y_pred = y_pred[:, :, :, 0:1]
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    tp = K.sum(
        K.round(K.clip(y_true, 0, 1)) * K.round(K.clip(y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives

    recall = tp / (pp + 1e-8)
    return recall


def M_F1(y_true, y_pred):
    """F1-score"""
    precision = M_Precision(y_true, y_pred)
    recall = M_Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def M_IOU(y_true: tf.Tensor,
          y_pred: tf.Tensor):
    # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))
    y_pred = y_pred[:, :, :, 0:1]
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    max_pool_2d = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    predict = K.round(K.clip(y_pred, 0, 1))
    Intersection = K.sum(
        K.round(K.clip(y_true_max, 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true_max, 0, 1)) * predict) + \
            (K.sum(K.round(K.clip(y_true, 0, 1))) - K.sum(K.round(K.clip(y_true, 0, 1)) *
                                                          K.round(K.clip(y_pred, 0, 1)))) + \
            (K.sum(K.round(K.clip(y_pred, 0, 1))) - K.sum(K.round(K.clip(y_true_max, 0, 1)) *
                                                          K.round(K.clip(y_pred, 0, 1))))
    iou = Intersection / (Union + 1e-8)

    return iou


class M_Precision_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """精确率"""
        # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))

        # y_pred = y_pred[:, :, :, 0:1]

        # 调整计算方式，使其计算每个类别的精确率

        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        y_pred = y_pred[:, :, :, 1:]
        y_true = y_true[-1:, :, :, 1:]

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)
        # true positives

        tp_crack = K.sum(K.round(K.round(K.clip(y_pred, 0, 1)) * K.round(K.clip(y_true_max, 0, 1))))
        tp_background =  K.sum(K.round(K.round(K.clip((1 - y_pred), 0, 1)) * K.round(K.clip((1 - y_true_max), 0, 1))))

        pp_crack = K.sum(K.round(K.clip(y_pred, 0, 1)))
        pp_background = K.sum(K.round(K.clip((1 - y_pred), 0, 1)))  # predicted positives

        precision = 0.5 * ((tp_crack + 1e-8) / (pp_crack + 1e-8) + (tp_background + 1e-8) / (pp_background + 1e-8))

        return precision


class M_Recall_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """召回率"""
        # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))

        # y_pred = y_pred[:, :, :, 0:1]

        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        y_pred = y_pred[:, :, :, 1:]
        y_true = y_true[-1:, :, :, 1:]

        max_pool_2d = tf.keras.layers.MaxPooling2D(
            pool_size=(1, 1), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)
        # true positives
        tp_crack = K.sum(K.round(K.round(K.clip(y_pred, 0, 1)) * K.round(K.clip(y_true_max, 0, 1))))
        tp_background = K.sum(K.round(K.round(K.clip((1 - y_pred), 0, 1)) * K.round(K.clip((1 - y_true_max), 0, 1))))
        pp_crack = K.sum(K.round(K.clip(y_true, 0, 1)))
        pp_background = K.sum(K.round(K.clip((1 - y_true), 0, 1)))  # predicted positives

        recall = 0.5 * ((tp_crack + 1e-8) / (pp_crack + 1e-8) + (tp_background + 1e-8) / (pp_background + 1e-8))

        return recall


class M_F1_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """F1"""
        # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))
        # y_pred = y_pred[:, :, :, 0:1]
        # y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)
        # max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
        # y_true_max = max_pool_2d(y_true)
        # # true positives
        # tp = K.sum(K.round(K.round(K.clip(y_pred, 0, 1)) * K.round(K.clip(y_true_max, 0, 1))))
        # pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
        # precision = tp / (pp + 1e-8)
        # pp = K.sum(K.round(K.clip(y_true_max, 0, 1)))  # predicted positives
        # recall = tp / (pp + 1e-8)
        # f1 = 2 * precision * recall / (precision + recall + 1e-8)

        precision = M_Precision_class(self.threshold)(y_true, y_pred)
        recall = M_Recall_class(self.threshold)(y_true, y_pred)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1


class M_IOU_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        # y_pred = K.softmax(K.concatenate([y_pred * 5, (1 - y_pred) * 5], axis=-1))
        # y_pred = y_pred[:, :, :, 0:1]
        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        y_pred = y_pred[:, :, :, 1:]
        y_true = y_true[-1:, :, :, 1:]

        max_pool_2d = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)
        predict = K.round(K.clip(y_pred, 0, 1))
        Intersection_crack = K.sum(
            K.round(K.clip(y_true_max, 0, 1)) * predict)
        Union_crack = K.sum(K.round(K.clip(y_true_max, 0, 1)) * predict) + \
                      (K.sum(K.round(K.clip(y_true, 0, 1))) - K.sum(K.round(K.clip(y_true, 0, 1)) *
                                                                    K.round(K.clip(y_pred, 0, 1)))) + \
                      (K.sum(K.round(K.clip(y_pred, 0, 1))) - K.sum(K.round(K.clip(y_true_max, 0, 1)) *
                                                                    K.round(K.clip(y_pred, 0, 1))))
        iou_crack = (Intersection_crack + 1e-8) / (Union_crack + 1e-8)

        predict = K.round(K.clip((1 - y_pred), 0, 1))
        Intersection_background = K.sum(
            K.round(K.clip((1 - y_true_max), 0, 1)) * predict)
        Union_background = K.sum(K.round(K.clip((1 - y_true_max), 0, 1)) * predict) + \
                           (K.sum(K.round(K.clip((1 - y_true), 0, 1))) - K.sum(K.round(K.clip((1 - y_true), 0, 1)) *
                                                                               K.round(K.clip((1 - y_pred), 0, 1)))) + \
                           (K.sum(K.round(K.clip((1 - y_pred), 0, 1))) - K.sum(K.round(K.clip((1 - y_true_max), 0, 1)) *
                                                                               K.round(K.clip((1 - y_pred), 0, 1))))
        iou_background = (Intersection_background + 1e-8) / \
                         (Union_background + 1e-8)

        # return iou_crack
        return 0.5 * (iou_crack + iou_background)


METRICS = [
    keras.metrics.binary_accuracy,
    M_Precision,
    M_Recall,
    M_F1,
    M_IOU
]
