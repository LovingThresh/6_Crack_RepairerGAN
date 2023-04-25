# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 16:38
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : Model_Comparison.py
# @Software: PyCharm
# @Description: 模型比较
import os

import numpy as np
import tf_slim
# 获取两种tensorflow的模型架构
# 从model_dir获取相应网络的权重
# 读取分割数据集
# 载入模型与权重,调用evaluate进行测试

from matplotlib import pyplot as plt

import pylib as py
from I_data import *
from Metrics import *
import Cycle_data as data
from CRFs import CRFs_array
from Metrics_compara import *

hyper_params = {
    'ex_number': 'A2A_Up_G_MCFF_RTX4090',
    'device': '3080Ti',
    'data_type': 'MCFF_crack',
    'datasets_dir': 'P:/GAN/CycleGAN-liuye-master/RepairerGAN/datasets',
    'load_size': 448,
    'crop_size': 448,
    'batch_size': 1,
    'epochs': 5,
    'epoch_decay': 4,
    'learning_rate_G': 0.0002,
    'learning_rate_D': 0.00002,
    'learning_rate_D_B_weight': 1.0,
    'beta_1': 0.5,
    'adversarial_loss_mode': 'lsgan',
    'gradient_penalty_mode': 'none',
    'gradient_penalty_weight': 10.0,
    'g_loss_weight': 2.0,
    'cycle_loss_weight': 10.0,
    'identity_loss_weight': 10.0,
    'ssim_loss_weight': 1.0,
    'ssim_Fake_True_weight': 5.0,
    'std_loss_weight': 50.0,
    'pool_size': 50,
    'lambda_reg': 1e-6,
    'starting_rate': 0.01
}

# Dataset
# Segmentation数据制作

A_img_val_paths = py.glob(py.join(
    hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive_4_500'), '*.jpg')
A_mask_paths = py.glob(py.join(
    hyper_params['datasets_dir'], hyper_params['data_type'], 'test_Positive_mask_4_500'), '*.jpg')
A_mask_dataset, len_mask_dataset = data.make_zip_dataset(A_img_val_paths, A_mask_paths,
                                                         hyper_params['batch_size'],
                                                         hyper_params['load_size'],
                                                         hyper_params['crop_size'],
                                                         training=False,
                                                         shuffle=False,
                                                         repeat=1,
                                                         random_fn=False,
                                                         mask=True)
batch_size = hyper_params['batch_size']
data_path = 'P:/GAN/CycleGAN-liuye-master/RepairerGAN/datasets/MCFF_crack'

# train_lines, num_train = get_data(path='{}/train.txt'.format(data_path), training=False)
# validation_lines, num_val = get_data(path='{}/val.txt'.format(data_path), training=False)
test_lines, num_test = get_data(
    path='{}/test_DeepCrack.txt'.format(data_path), training=False)
data_type = 'test_Positive_mask_5_DeepCrack'
# train_dataset = get_dataset_label(train_lines, batch_size,
#                                   A_img_paths='{}/train_Positive/'.format(data_path),
#                                   B_img_paths='{}/{}/'.format(data_path, data_type),
#                                   shuffle=True,
#                                   KD=False,
#                                   training=True,
#                                   Augmentation=True)
# validation_dataset = get_dataset_label(validation_lines, batch_size,
#                                        A_img_paths='{}/val_Positive/'.format(data_path),
#                                        B_img_paths='{}/{}/'.format(data_path, data_type),
#                                        shuffle=False,
#                                        KD=False,
#                                        training=False,
#                                        Augmentation=False)
test_dataset = get_dataset_label(test_lines, batch_size,
                                 A_img_paths='{}/test_Positive_5_DeepCrack/'.format(
                                     data_path),
                                 B_img_paths='{}/{}/'.format(data_path,
                                                             data_type),
                                 shuffle=False,
                                 KD=False,
                                 training=False,
                                 Augmentation=False)

model_dict = \
    {
        'crack':
            {
                'crack_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-25-21-12-31.713630/checkpoint/ep002-val_loss504.255',
                'Grad_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-10-30-17.680163/checkpoint/ep007-val_loss514.432',
                'Grad_CAM++': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-14-16-55.630579/checkpoint/ep001-val_loss429.734',
                'Score_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-15-47-15.393470/checkpoint/ep002-val_loss270.805',
                'AblationCAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-18-58-13.580796/checkpoint/ep005-val_loss448.592'},
        'MCFF_crack':
            {
                'crack_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-22-09-27.755814-Repeat/checkpoint/ep004-val_loss715.166',
                'Grad_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-23-19-09.546718/checkpoint/ep008-val_loss700.309',
                'Grad_CAM++': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-27-23-34-29.862276/checkpoint/ep006-val_loss706.755',
                'Score_CAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-28-00-20-35.135887/checkpoint/ep006-val_loss484.614',
                'AblationCAM': 'M:/CycleGAN(WSSS)/File/checpoint/output/2022-08-28-00-35-43.121206/checkpoint/ep004-val_loss743.157'},
        'RepairerGAN_crack':
            {'RepairerGAN': 'M:/CycleGAN(WSSS)/GAN_Checkpoint/SSIM_ConvNext/0-2700-0.8056701421737671'},
        'RepairerGAN_MCFF':
            {'RepairerGAN': 'M:/CycleGAN(WSSS)/File/checpoint/CycleGAN/4-16500-0.501158595085144'}
    }


def Pr_and_Re_curve(dataset, tf_model):
    """
    Receive the TensorFlow model, dataset, return precision and recall dict based different thresholds
    Parameters
    ----------
    dataset: dataset is a list of tuples, each tuple contains an image and a mask
    tf_model: tensorflow model

    Returns precision and recall dict base on different thresholds
    ---------
    """
    initial_learning_rate = 5e-5
    optimizer = keras.optimizers.RMSprop(initial_learning_rate)
    # tf_model = keras.models.Model(inputs=tf_model.inputs, outputs=tf_model.outputs[0][:, :, :, 0:1])

    # 创建一个字典metrics_dict，字典的键为i，值为pr与re
    metrics_dict = {}
    for i in range(0, 21, 1):
        tf_model.compile(optimizer=optimizer,
                         loss=keras.losses.BinaryCrossentropy(),
                         metrics=['accuracy',
                                  # A_Precision, A_Recall, A_F1, A_IOU,
                                  M_Precision_class(threshold=i / 20.),
                                  M_F1_class(threshold=i / 20.),
                                  M_Recall_class(threshold=i / 20.),
                                  M_IOU_class(threshold=i / 20.)])

        predict_dict = tf_model.evaluate(dataset, steps=39, return_dict=True)
        # 下述形式以字典的形式保存
        metrics_dict[i] = {'pr': predict_dict['m__precision_class'],
                           're': predict_dict['m__recall_class'],
                           'f1': predict_dict['m_f1_class'],
                           'iou': predict_dict['m_iou_class']}

    return metrics_dict


# 以model_dict['crack']['crack_CAM']模型为例
# 保存字典为crack_CAM.npy, 路径前缀为'M:/CycleGAN(WSSS)/Comparison/dict'
def save_pr_and_re_instance():
    model_dir = model_dict['RepairerGAN_MCFF']['RepairerGAN']
    model = keras.models.load_model(filepath=model_dir,
                                    # custom_objects={'A_IOU': A_IOU,
                                    #                 'A_Precision': A_Precision,
                                    #                 'A_Recall': A_Recall,
                                    #                 'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss}
                                    custom_objects={'M_IOU': M_IOU,
                                                    'M_Precision': M_Precision,
                                                    'M_Recall': M_Recall,
                                                    'M_F1': M_F1}
                                    )

    metrics_dict_instance = Pr_and_Re_curve(test_dataset, model)
    # metrics_dict_instance = Pr_and_Re_curve(A_mask_dataset, model)
    np.save('M:/CycleGAN(WSSS)/Comparison/dict/CAM_Crack500_MIoU_Revise.npy',
            metrics_dict_instance)


# save_pr_and_re_instance()


def save_pr_and_re_all():
    metrics_dict_instance = {}
    for i in ['MCFF_crack']:
        for j in model_dict[i].keys():
            model_dir = model_dict[i][j]
            model = keras.models.load_model(filepath=model_dir,
                                            custom_objects={'A_IOU': A_IOU,
                                                            'A_Precision': A_Precision,
                                                            'A_Recall': A_Recall,
                                                            'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss}
                                            )

            metrics_dict_instance[j] = Pr_and_Re_curve(test_dataset, model)
        np.save('M:/CycleGAN(WSSS)/Comparison/dict/DeepCrack_MIoU' +
                i + '.npy', metrics_dict_instance)


save_pr_and_re_all()


# 读取字典metrics_dict_instance
# metrics_dict_instance的结构为{'model_name': {'threshold':{'pr': float, 're': float}}}
# 绘制成pr-re曲线


def plot_pr_and_re_curve(save_dict_path):
    """
    Receive the metrics_dict_instance, model_name, plot the pr-re curve
    Parameters
    ----------
    save_dict_path: path of dict {'model_name': {'threshold':{'pr': float, 're': float}}}

    Returns pr-re curve
    ---------
    """
    metrics_dict = np.load(save_dict_path, allow_pickle=True).item()
    # metrics_dict likes {'model_name': {'threshold':{'pr': float, 're': float}}}
    plt.figure(figsize=(10, 8))
    # plt.xlim(0.3, 1.05)
    # plt.ylim(-0.04, 1.1)
    plt.title('Pr-Re Curve', fontsize=20)
    plt.ylabel('Precision', fontsize=15)
    plt.xlabel('Recall', fontsize=15)
    # 画一条x=y的浅色直线, 作为参考线,在线上写上文字‘Pr=Re’
    plt.plot([0.25, 0.85], [0.25, 0.85], color='lightgray',
             linestyle='--', linewidth=4)
    plt.text(0.6, 0.5, 'Pr=Re', fontsize=20, color='gray')
    for i in metrics_dict.keys():
        # the key i means model_name, like 'crack_CAM'
        # different model has different pr-re line labels and soft colors
        if i == 'crack_CAM':
            label = 'Crack-CAM'
            color = '#005f73'
        elif i == 'Grad_CAM':
            label = 'Grad-CAM'
            color = '#0a9396'
        elif i == 'Grad_CAM++':
            label = 'Grad-CAM++'
            color = '#94d2bd'
        elif i == 'Score_CAM':
            label = 'Score-CAM'
            color = '#e9d8a6'
        elif i == 'AblationCAM':
            label = 'Ablation-CAM'
            color = '#ee9b00'
        elif i == 'RepairerGAN':
            label = 'RepairerGAN'
            color = '#ca6702'
        else:
            label = 'None'
            color = 'black'
        x, y = [], []
        for j in metrics_dict[i].keys():
            # the key j means threshold, like 0.5
            y.append(metrics_dict[i][j]['pr'])
            x.append(metrics_dict[i][j]['re'])
        plt.plot(x, y, label=label, color=color,
                 linewidth=2, marker='o', markersize=5)
        plt.legend(loc='lower left', fontsize=12)
    # plt.show()
    # 以高质量保存图片
    plt.savefig('M:/CycleGAN(WSSS)/Comparison/PR-RE_Curve_MCFF_crack.png',
                dpi=600, bbox_inches='tight')


# path = 'M:/CycleGAN(WSSS)/Comparison/dict/crack_combined_new.npy'
# plot_pr_and_re_curve(path)

# path = 'M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack.npy'
# plot_pr_and_re_curve(path)

# path = 'M:/CycleGAN(WSSS)/Comparison/dict/crack.npy'
# plot_pr_and_re_curve(path)


def save_instance():
    model_dir = model_dict['RepairerGAN_MCFF']['RepairerGAN']
    model = keras.models.load_model(filepath=model_dir,
                                    custom_objects={'M_IOU': M_IOU,
                                                    'M_Precision': M_Precision,
                                                    'M_Recall': M_Recall,
                                                    'M_F1': M_F1,
                                                    # 'Asymmetry_Binary_Loss': Asymmetry_Binary_Loss
                                                    }
                                    )
    model = keras.models.Model(inputs=[model.inputs], outputs=[model.outputs])
    # 对模型设置不同的阈值，[0, 20, 1]
    metrics_dict = {}
    for i in range(0, 21, 1):
        # for i in range(1, 5, 1):
        initial_learning_rate = 5e-5
        optimizer = keras.optimizers.RMSprop(initial_learning_rate)
        model.compile(optimizer=optimizer,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', M_Precision_class(threshold=i / 20.),
                               M_F1_class(threshold=i / 20.),
                               M_Recall_class(threshold=i / 20.),
                               M_IOU_class(threshold=i / 20.)])
        # 使用np.array进行储存以下信息: precision, recall, f1, iou, precision_crf, recall_crf, f1_crf, iou_crf
        precision_list = np.array([])
        recall_list = np.array([])
        f1_list = np.array([])
        iou_list = np.array([])

        precision_crf_list = np.array([])
        recall_crf_list = np.array([])
        f1_crf_list = np.array([])
        iou_crf_list = np.array([])

        for j in A_mask_dataset:
            # for j in test_dataset:
            predict_data = model.predict(j[0])
            predict_array = predict_data[0]

            image_array = np.uint8(
                np.clip(j[0] * 127.5 + 127.5, 0, 255)).reshape(448, 448, 3)

            predict_array = np.uint8(
                predict_array >= (i / 20)).reshape(448, 448)
            # predict_array = np.uint8(predict_array >= 0.3).reshape(224, 224)
            CRFs_predict_array = CRFs_array(image_array, predict_array * 255)

            # 绘制predict
            # plt.figure(figsize=(10, 8))
            # plt.subplot(1, 2, 1)
            # plt.imshow(predict_array * 127)
            # plt.title('predict')
            # # 绘制CRFs_predict
            # plt.subplot(1, 2, 2)
            # plt.imshow(CRFs_predict_array * 127)
            # plt.title('CRFs_predict')
            # # plt.show()
            #
            # plt.savefig('a.png', dpi=600, bbox_inches='tight')
            # 将predict_array和CRFs_predict_array转换为TensorFlow Tensor
            predict_array = tf.convert_to_tensor(predict_array)
            CRFs_predict_array = tf.cast(tf.convert_to_tensor(
                CRFs_predict_array), dtype=tf.float32)
            # 与j进行Pr, Re, IoU的计算
            y_true = tf.reshape(
                tf.cast(j[1], dtype=tf.float32), (448, 448)) / 255.
            max_pool_2d = tf.keras.layers.MaxPooling2D(
                pool_size=(5, 5), strides=(1, 1), padding='same')
            y_true_max = max_pool_2d(tf.reshape(y_true, (1, 448, 448, 1)))
            y_true_max = tf.reshape(y_true_max, (448, 448))

            predict_array = tf.cast(predict_array, dtype=tf.float32)
            CRFs_predict_array = tf.cast(CRFs_predict_array, dtype=tf.float32)

            tp_pred = y_true_max * predict_array
            pp_pred = predict_array
            tp_pred_crf = y_true_max * CRFs_predict_array
            pp_pred_crf = CRFs_predict_array
            precision = (tf.reduce_sum(tp_pred) + 1e-8) / \
                (tf.reduce_sum(pp_pred) + 1e-8)
            precision_crf = (tf.reduce_sum(tp_pred_crf) + 1e-8) / \
                (tf.reduce_sum(pp_pred_crf) + 1e-8)

            tp_pred = y_true * predict_array
            pp_pred = y_true
            tp_pred_crf = y_true * CRFs_predict_array
            pp_pred_crf = y_true
            recall = (tf.reduce_sum(tp_pred) + 1e-8) / \
                (tf.reduce_sum(pp_pred) + 1e-8)
            recall_crf = (tf.reduce_sum(tp_pred_crf) + 1e-8) / \
                (tf.reduce_sum(pp_pred_crf) + 1e-8)

            f1 = 2 * precision * recall / (precision + recall)
            f1_crf = 2 * precision_crf * recall_crf / \
                (precision_crf + recall_crf)

            # tp_pred = y_true_max * predict_array
            Intersection_pred = K.sum(
                K.round(K.clip(y_true_max, 0, 1)) * predict_array)
            Union_pred = K.sum(K.round(K.clip(y_true_max, 0, 1)) * predict_array) + \
                (K.sum(K.round(K.clip(y_true, 0, 1))) - K.sum(K.round(K.clip(y_true, 0, 1)) *
                                                              K.round(K.clip(predict_array, 0, 1)))) + \
                (K.sum(K.round(K.clip(predict_array, 0, 1))) - K.sum(K.round(K.clip(y_true_max, 0, 1)) *
                                                                     K.round(K.clip(predict_array, 0, 1))))
            # iou = (tf.reduce_sum(tp_pred) + 1e-8) / (tf.reduce_sum(y_true) + tf.reduce_sum(predict_array) - tf.reduce_sum(tp_pred) + 1e-8)
            iou = (Intersection_pred + 1e-8) / (Union_pred + 1e-8)
            Intersection_pred_crf = K.sum(
                K.round(K.clip(y_true_max, 0, 1)) * CRFs_predict_array)
            Union_pred_crf = K.sum(K.round(K.clip(y_true_max, 0, 1)) * CRFs_predict_array) + \
                (K.sum(K.round(K.clip(y_true, 0, 1))) - K.sum(K.round(K.clip(y_true, 0, 1)) *
                                                              K.round(K.clip(CRFs_predict_array, 0, 1)))) + \
                (K.sum(K.round(K.clip(CRFs_predict_array, 0, 1))) - K.sum(
                    K.round(K.clip(y_true_max, 0, 1)) *
                    K.round(K.clip(CRFs_predict_array, 0, 1))))
            iou_crf = (Intersection_pred_crf + 1e-8) / (Union_pred_crf + 1e-8)

            # 将相应的信息保存至array中
            precision_list = np.append(precision_list, precision.numpy())
            recall_list = np.append(recall_list, recall.numpy())
            f1_list = np.append(f1_list, f1.numpy())
            iou_list = np.append(iou_list, iou.numpy())

            precision_crf_list = np.append(
                precision_crf_list, precision_crf.numpy())
            recall_crf_list = np.append(recall_crf_list, recall_crf.numpy())
            f1_crf_list = np.append(f1_crf_list, f1_crf.numpy())
            iou_crf_list = np.append(iou_crf_list, iou_crf.numpy())

        # 将这些array的平均值打印出来
        print('threshold: ', i / 20.)
        print('precision: ', np.mean(precision_list))
        print('recall: ', np.mean(recall_list))
        print('f1: ', np.mean(f1_list))
        print('iou: ', np.mean(iou_list))
        print('precision_crf: ', np.mean(precision_crf_list))
        print('recall_crf: ', np.mean(recall_crf_list))
        print('f1_crf: ', np.mean(f1_crf_list))
        print('iou_crf: ', np.mean(iou_crf_list))
        print('-----------------------------------')
        # 将这些array的平均值保存至字典metrics_dict中
        metrics_dict[i] = {'pr': np.mean(precision_list),
                           're': np.mean(recall_list),
                           'f1': np.mean(f1_list),
                           'iou': np.mean(iou_list),
                           'pr_crf': np.mean(precision_crf_list),
                           're_crf': np.mean(recall_crf_list),
                           'f1_crf': np.mean(f1_crf_list),
                           'iou_crf': np.mean(iou_crf_list)}
        # 强制规定当i为0时，pr_crf为pr，re_crf为re，f1_crf为f1，iou_crf为iou
        if i == 0:
            metrics_dict[i]['pr_crf'] = metrics_dict[i]['pr']
            metrics_dict[i]['re_crf'] = metrics_dict[i]['re']
            metrics_dict[i]['f1_crf'] = metrics_dict[i]['f1']
            metrics_dict[i]['iou_crf'] = metrics_dict[i]['iou']
        # 强制规定当i为20时，pr, re, f1, iou的值为1, 0, 0, 0
        if i == 20:
            metrics_dict[i]['pr'] = np.array(1.)
            metrics_dict[i]['re'] = np.array(0.)
            metrics_dict[i]['f1'] = np.array(0.)
            metrics_dict[i]['iou'] = np.array(0.)
            metrics_dict[i]['pr_crf'] = np.array(1.)
            metrics_dict[i]['re_crf'] = np.array(0.)
            metrics_dict[i]['f1_crf'] = np.array(0.)
            metrics_dict[i]['iou_crf'] = np.array(0.)

    # 将metrics_dict保存至npy文件中
    np.save('M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack_RepairerGAN.npy', metrics_dict)


# save_instance()


# 加载crack.npy,给每个模型的阈值增加一个额外的键20,设置其pr, re, iou, f1的值
def add_data_crack(father_path):
    father_dict: dict = np.load(father_path, allow_pickle=True).item()
    for i in father_dict.keys():
        father_dict[i]['20'] = {}
        father_dict[i]['20']['pr'] = 1.
        father_dict[i]['20']['re'] = 0.
        father_dict[i]['20']['f1'] = 0.
        father_dict[i]['20']['iou'] = 0.
    np.save(father_path, father_dict)


# 加载MCFF_crack_RepairerGAN_crf.npy文件,将其与crack.npy进行合并
def insert_RepairerGAN_dict(RepairerGAN_path, father_path):
    # 加载RepairerGAN.npy文件
    RepairerGAN_dict = np.load(RepairerGAN_path, allow_pickle=True).item()
    # 加载father.npy文件
    father_dict = np.load(father_path, allow_pickle=True).item()
    # 将RepairerGAN_dict中的值插入到father_dict中
    father_dict['RepairerGAN'] = {}
    for i in range(0, 21, 1):
        father_dict['RepairerGAN'][i] = {}
        father_dict['RepairerGAN'][i]['pr'] = RepairerGAN_dict[i]['pr']
        father_dict['RepairerGAN'][i]['re'] = RepairerGAN_dict[i]['re']
        father_dict['RepairerGAN'][i]['f1'] = RepairerGAN_dict[i]['f1']
        father_dict['RepairerGAN'][i]['iou'] = RepairerGAN_dict[i]['iou']
    # 将father_dict保存至npy文件中
    np.save('M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack_combined.npy', father_dict)


# RepairerGAN_path, father_path = 'M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack_RepairerGAN_++.npy', \
#                                 'M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack.npy'
# insert_RepairerGAN_dict(RepairerGAN_path, father_path)

#
# path = 'M:/CycleGAN(WSSS)/Comparison/dict/MCFF_crack_combined.npy'
# plot_pr_and_re_curve(path)
# 实现排序算法
