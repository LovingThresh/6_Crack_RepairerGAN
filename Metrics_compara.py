import tensorflow.keras.losses
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def Asymmetry_Binary_Loss_2(y_true, y_pred):
    # 想要损失函数更加关心裂缝的标签值1
    # y_true_0, y_pred_0 = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    # # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    # y_true_1, y_pred_1 = y_true[:, :, :, 1], y_pred[:, :, :, 1]
    bcr = tf.losses.binary_crossentropy
    # return bcr(y_true_0, y_pred_0) + bcr(y_true_1, y_pred_1)
    return bcr(y_true, y_pred)


def Constraints_Loss(y_true, y_pred):
    y_true = tf.ones_like(y_true[:, :, :, 0]) * 10
    y_pred_0 = y_pred[:, :, :, 0] * 10
    y_pred_1 = y_pred[:, :, :, 1] * 10
    y_pred = y_pred_0 + y_pred_1
    mse = tf.losses.mean_squared_error

    return mse(y_true, y_pred)


def dice_loss(y_true, y_pred, ep=1e-8):
    ep = tf.constant(ep, tf.float32)
    alpha = tf.constant(2, tf.float32)
    # y_true_0, y_pred_0 = tf.cast(y_true[:, :, :, 0], tf.float32), tf.cast(y_pred[:, :, :, 0].as_dtype(tf.float32),
    # tf.float32)
    # y_true_0, y_pred_0 = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    y_true_0, y_pred_0 = y_true, y_pred
    intersection = alpha * tf.cast(K.sum(y_pred_0 * y_true_0), tf.float32) + ep
    union = tf.cast(K.sum(y_pred_0), tf.float32) + tf.cast(K.sum(y_true_0), tf.float32) + ep
    loss = 1 - intersection / union

    return loss


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # y_true与y_pred都是矩阵！（Unet）
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * keras.losses.binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def Total_loss(y_true, y_pred):
    return Asymmetry_Binary_Loss(y_true, y_pred) + Constraints_Loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# KD损失函数-alpha=0.9
def S_KD_Loss(y_true, y_pred, alpha=0.9):
    soft_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return alpha * soft_label_loss


def H_KD_Loss(y_true, y_pred, alpha=0.9):
    hard_label_loss = Asymmetry_Binary_Loss(y_true, y_pred)

    return (1 - alpha) * hard_label_loss


def M_Precision_1(y_true, y_pred):
    """精确率"""
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3, dtype=tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    # true positives
    tp = K.sum(K.round(K.round(K.clip(y_pred[:, :, :, 1], 0, 1)) * K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1))))
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # predicted positives
    precision = tp / (pp + 1e-8)
    return precision


# 只看核心区域
def M_Recall_1(y_true, y_pred):
    """召回率"""

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3, dtype=tf.float32)

    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)))  # possible positives

    recall = tp / (pp + 1e-8)
    return recall


def M_F1_1(y_true, y_pred):
    """F1-score"""
    precision = M_Precision_1(y_true, y_pred)
    recall = M_Recall_1(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def M_IOU_0(y_true: tf.Tensor,
            y_pred: tf.Tensor):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3, dtype=tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    predict = K.round(K.clip(y_pred[:, :, :, 0], 0, 1))
    Intersection = K.sum(
        K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) * predict) + \
            (K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1))) - K.sum(K.round(K.clip(y_true[-1:, :, :, 0], 0, 1)) *
                                                                        K.round(K.clip(y_pred[:, :, :, 0], 0, 1)))) + \
            (K.sum(K.round(K.clip(y_pred[:, :, :, 0], 0, 1))) - K.sum(K.round(K.clip(y_true_max[-1:, :, :, 0], 0, 1)) *
                                                                      K.round(K.clip(y_pred[:, :, :, 0], 0, 1))))
    iou = Intersection / (Union + 1e-8)

    return iou


def M_IOU_1(y_true: tf.Tensor,
            y_pred: tf.Tensor):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3, dtype=tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    predict = K.round(K.clip(y_pred[:, :, :, 1], 0, 1))
    Intersection = K.sum(
        K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * predict) + \
            (K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1))) - K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) *
                                                                        K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))) + \
            (K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1))) - K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) *
                                                                      K.round(K.clip(y_pred[:, :, :, 1], 0, 1))))
    iou = Intersection / (Union + 1e-8)

    return iou


def M_IOU_2(y_true: tf.Tensor,
            y_pred: tf.Tensor):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=3, dtype=tf.float32)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
    y_true_max = max_pool_2d(y_true)
    predict = K.round(K.clip(y_pred[:, :, :, 2], 0, 1))
    Intersection = K.sum(
        K.round(K.clip(y_true_max[-1:, :, :, 2], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 2], 0, 1)) * predict) + \
            (K.sum(K.round(K.clip(y_true[-1:, :, :, 2], 0, 1))) - K.sum(K.round(K.clip(y_true[-1:, :, :, 2], 0, 1)) *
                                                                        K.round(K.clip(y_pred[:, :, :, 2], 0, 1)))) + \
            (K.sum(K.round(K.clip(y_pred[:, :, :, 2], 0, 1))) - K.sum(K.round(K.clip(y_true_max[-1:, :, :, 2], 0, 1)) *
                                                                      K.round(K.clip(y_pred[:, :, :, 2], 0, 1))))
    iou = Intersection / (Union + 1e-8)

    return iou


def A_Precision(y_true, y_pred):
    """精确率"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)

    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # predicted positives
    precision = (tp + 1e-8) / (pp + 1e-8)
    return precision


def A_Recall(y_true, y_pred):
    """召回率"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    tp = K.sum(
        K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) * K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)))  # possible positives

    recall = (tp + 1e-8) / (pp + 1e-8)
    return recall


def A_F1(y_true, y_pred):
    """F1-score"""
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    precision = A_Precision(y_true, y_pred)
    recall = A_Recall(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def A_IOU(y_true: tf.Tensor,
          y_pred: tf.Tensor):
    y_pred = tf.cast(y_pred > tf.constant(0.3), tf.float32)
    predict = K.round(K.clip(y_pred[:, :, :, 1], 0, 1))
    Intersection = K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) * predict)
    Union = K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) + predict)
    iou = (Intersection + 1e-8) / (Union - Intersection + 1e-8)
    return iou


def iou_keras(y_true, y_pred):
    """
    Return the Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())

    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou_keras(y_true, y_pred):
    """
    Return the mean Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the mean IoU
    """
    y_true = y_true[-1:, :, :, 0]
    y_pred = y_pred[:, :, :, 0]
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())

    mean_iou = 0.

    thre_list = list(np.arange(0.0000001, 0.99, 0.05))

    for thre in thre_list:
        y_pred_temp = K.cast(y_pred >= thre, K.floatx())
        y_pred_temp = K.cast(K.equal(y_pred_temp, label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred_temp)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred_temp) - intersection
        iou = K.switch(K.equal(union, 0), 1.0, intersection / union)
        mean_iou = mean_iou + iou

    return mean_iou / len(thre_list)


def Binary_Focal_loss(gamma=2, alpha=0.25):
    # alpha = tf.constant(alpha, dtype=np.float32)
    # gamma = tf.constant(gamma, dtype=np.float32)

    def binary_focal_loss(y_true, y_pred):
        y_true = y_true[:, :, :, 0]
        y_pred = y_pred[:, :, :, 0]

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = -alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

        return K.sum(focal_loss, axis=-1)

    return binary_focal_loss


# 自定义损失函数
def Asymmetry_Binary_Loss(y_true, y_pred, alpha=200):
    # 纯净状态下alpha为1
    # 想要损失函数更加关心裂缝的标签值1
    alpha = 100
    y_true_0, y_pred_0 = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    # y_true_0, y_pred_0 = y_true[:, :, :, 0] * 255, y_pred[:, :, :, 0] * 255
    y_true_1, y_pred_1 = y_true[:, :, :, 1] * alpha, y_pred[:, :, :, 1] * alpha
    mse = tf.losses.mean_squared_error
    return mse(y_true_0, y_pred_0) + mse(y_true_1, y_pred_1)


class A_Precision_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """精确率"""
        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)

        tp = K.sum(
            K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * K.round(
                K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
        pp = K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # predicted positives
        precision = (tp + 1e-8) / (pp + 1e-8)

        return precision


class A_Recall_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """召回率"""
        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)

        tp = K.sum(
            K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * K.round(
                K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
        pp = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)))  # predicted positives

        recall = (tp + 1e-8) / (pp + 1e-8)

        return recall


class A_F1_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """F1"""

        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)

        tp = K.sum(
            K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * K.round(
                K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
        pp = K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1)))  # predicted positives
        precision = (tp + 1e-8) / (pp + 1e-8)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)

        tp = K.sum(
            K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * K.round(
                K.clip(y_pred[:, :, :, 1], 0, 1)))  # true positives
        pp = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)))  # predicted positives
        recall = (tp + 1e-8) / (pp + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1


class A_IOU_class:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        """IoU"""
        y_pred = tf.cast(y_pred > tf.constant(self.threshold), tf.float32)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')
        y_true_max = max_pool_2d(y_true)

        predict = K.round(K.clip(y_pred[:, :, :, 1], 0, 1))
        Intersection = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * predict)

        Union = K.sum(K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) * predict) + \
                (K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1))) - K.sum(
                    K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) *
                    K.round(K.clip(y_pred, 0, 1)))) + \
                (K.sum(K.round(K.clip(y_pred[:, :, :, 1], 0, 1))) - K.sum(
                    K.round(K.clip(y_true_max[-1:, :, :, 1], 0, 1)) *
                    K.round(K.clip(y_pred, 0, 1))))

        # Union = K.sum(K.round(K.clip(y_true[-1:, :, :, 1], 0, 1)) + predict)

        iou = (Intersection + 1e-8) / (Union - Intersection + 1e-8)
        return iou
