import random

import numpy as np
import tensorflow as tf
import tf2lib as tl
from PIL import Image
import cv2

train_teacher_y_name = ''
Aug_image = object
Aug_label = object
seed = int


# 图像分割的map函数


def map_function_for_keras(data):
    image, label = data[0], data[1]
    a = random.randint(1, 600)
    image = image * 127.5 + 127.5

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_saturation(image, 0.2, 0.8)
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_hue(image, 0.5)
    image = tf.image.random_contrast(image, 0.2, 0.8)

    image = (image - 127.5) / 127.5

    label = tf.image.random_flip_left_right(label, seed=a)
    label = tf.image.random_flip_up_down(label, seed=a)
    label = np.array(label)

    return np.array(image), [label, label, label, label]


def to_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])
            # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size,
                     training, shuffle=False, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


def get_data(path=r'I:\Image Processing\train.txt',
             training=True, shuffle=True):
    """
    获取样本和标签对应的行：获取训练集和验证集的数量
    :return: lines： 样本和标签的对应行： [num_train, num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines -> [1.jpg;1.jpg\n', '10.jpg;10.png\n', ......]
    # .jpg:样本  ：  .jpg：标签

    with open(path, 'r') as f:
        lines = f.readlines()

    # 打乱行， 打乱数据有利于训练
    if shuffle:
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

    if training:
        # 切分训练样本， 90% 训练： 10% 验证
        num_val = int(len(lines) * 0.1)
        num_train = len(lines) - num_val
        return lines, num_train, num_val

    if not training:
        num_test = len(lines)
        return lines, num_test


# 注意train_HEYE是一样的,这里的C_img_paths是随便写的
def get_dataset_label(lines, batch_size,
                      A_img_paths=r'I:\Image Processing\Rebuild_Image_95/',
                      B_img_paths=r'I:\Image Processing\Mix_img\95\label/',
                      C_img_paths=r'I:\Image Processing\Mix_img\95\label/',
                      shuffle=True, KD=False, training=False, Augmentation=False):
    """
        生成器， 读取图片， 并对图片进行处理， 生成（样本，标签）
        :param Augmentation:
        :param training:
        :param C_img_paths:
        :param KD:
        :param shuffle:
        :param B_img_paths:
        :param A_img_paths:
        :param lines: 样本和标签的对应行
        :param batch_size: 一次处理的图片数
        :return:  返回（样本， 标签）
        """

    global train_teacher_y_name, seed
    numbers = len(lines)
    read_line = 0

    while True:

        x_train = []
        y_train = []
        y_teacher_train = []

        # 一次获取batch——size大小的数据

        for t in range(batch_size):
            if shuffle:
                np.random.shuffle(lines)

            # 1. 获取训练文件的名字
            train_x_name = lines[read_line].split(',')[0]

            # 根据图片名字读取图片
            img = cv2.imread(A_img_paths + train_x_name)
            # img = img.resize(size)
            img_array = np.array(img)
            # img_array = to_clahe(img_array)
            # img_array = cv2.resize(img_array, (224, 224))
            if img_array.shape == (227, 227, 3):
                img_array = cv2.resize(img_array, (224, 224))
            size = (img_array.shape[0], img_array.shape[1])
            # img_teacher_array = cv2.imread(C_img_paths + train_teacher_y_name, cv2.IMREAD_GRAYSCALE)
            img_array = img_array / 255.0  # 标准化
            img_array = img_array * 2 - 1
            x_train.append(img_array)

            # 2.1 获取训练样本标签的名字
            train_y_name = lines[read_line].split(',')[1].replace('\n', '')

            # 2.2 获取Teacher训练样本标签的名字,此处将文件的后缀.png改成.jpg就能转移到相应的Teacher_Label了
            if KD:
                train_teacher_y_name = lines[read_line].split(',')[0].replace('\n', '')[:-4] + '.jpg'

            # 根据图片名字读取图片
            img_array = cv2.imread(B_img_paths + train_y_name[:-4] + '.png')
            if img_array is None:
                img_array = cv2.imread(B_img_paths + train_y_name[:-4] + '.jpg')
            img_array = cv2.resize(img_array, size)
            if KD:
                img_teacher_array = cv2.imread(C_img_paths + train_teacher_y_name, cv2.IMREAD_GRAYSCALE)
            else:
                img_teacher_array = 0
            # img.show()
            # print(train_y_name)
            # img = img.resize(size)  # 改变图片大小 -> (227, 227)
            # img_array = np.array(img)
            # img_array = img[:, :, :2]
            # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

            # 生成标签， 标签的shape是（227， 227， class_numbers) = (227, 227, 2), 里面的值全是0
            labels = np.zeros((img_array.shape[0], img_array.shape[1], 2), np.int)

            # 下面将(224,224,3) => (224,224,2),不仅是通道数的变化，还有，
            # 原本背景和裂缝在一个通道里面，现在将斑马线和背景放在不同的通道里面。
            # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
            # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
            # 相当于合并的图层分层！！！！
            labels[:, :, 0] = (img_array[:, :, 0] == 0).astype(int).reshape(size)
            labels[:, :, 1] = (img_array[:, :, 1] != 0).astype(int).reshape(size)
            labels = labels.astype(np.float32)
            # labels[:, :, 0] = (img_array[:, :, 1] == 1).astype(int).reshape(size)
            # labels[:, :, 1] = (img_array[:, :, 1] != 1).astype(int).reshape(size)
            if KD:
                teacher_label = ((img_teacher_array - 127.5) / 127.5).astype(np.float32).reshape(448, 448, 1)
                teacher_label_opposite = 1 - teacher_label
                teacher_label = np.concatenate([teacher_label, teacher_label_opposite], axis=2)
            else:
                teacher_label = 0

            y_train.append(labels)

            if KD:
                y_teacher_train.append(teacher_label)

            # 遍历所有数据，记录现在所处的行， 读取完所有数据后，read_line=0,打乱重新开始
            read_line = (read_line + 1) % numbers

        if not KD:
            image, label = np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

            if training:

                if Augmentation:
                    seed = random.choice([0, 0, 0, 1, 2, 3, 4, 5, 6])
                if not Augmentation:
                    seed = random.choice([0, 0, 0])

                def DataAugmentation(row_image, row_label, D_seed=0):

                    global Aug_image, Aug_label
                    in_seed = np.random.randint(0, 6)

                    if D_seed == 0:
                        Aug_image = row_image
                        Aug_label = row_label

                    if D_seed == 1:
                        Aug_image = tf.image.random_flip_left_right(row_image, seed=in_seed)
                        Aug_label = tf.image.random_flip_left_right(row_label, seed=in_seed)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = np.array(np.reshape(Aug_label, row_label.shape))

                    if D_seed == 2:
                        Aug_image = tf.image.random_flip_up_down(row_image, seed=in_seed)
                        Aug_label = tf.image.random_flip_up_down(row_label, seed=in_seed)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = np.array(np.reshape(Aug_label, row_label.shape))

                    if D_seed == 3:
                        Aug_image = tf.image.random_saturation(row_image, 0.2, 0.8)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = row_label

                    if D_seed == 4:
                        Aug_image = tf.image.random_contrast(row_image, 0.2, 0.8)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = row_label

                    if D_seed == 5:
                        Aug_image = tf.image.random_brightness(row_image, 0.5)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = row_label

                    if D_seed == 6:
                        Aug_image = tf.image.random_hue(row_image, 0.5)
                        Aug_image = np.array(np.reshape(Aug_image, row_image.shape))
                        Aug_label = row_label

                    return Aug_image, Aug_label

                image, label = DataAugmentation(image, label, D_seed=seed)

                label = label.reshape((size[0], size[1], 2))
                data = image, np.asarray([label])

                yield data

            else:
                label = label.reshape((size[0], size[1], 2))
                data = image, np.asarray([label])

                yield data

            # yield np.array(x_train), np.array(y_train)
        if KD:
            yield np.array(x_train), [np.array(y_train), np.array(y_teacher_train)]


def get_test_dataset_label(lines,
                           A_img_paths=r'I:\Image Processing\Rebuild_Image_95/',
                           B_img_paths=r'I:\Image Processing\Mix_img\95\label/',
                           C_img_paths=r'I:\Image Processing\Mix_img\95\label/',
                           size=(512, 512),
                           KD=False):
    numbers = len(lines)

    x_train = []
    y_train = []
    y_teacher_train = []

    for read_line in range(numbers):
        train_x_name = lines[read_line].split(',')[0]

        # 根据图片名字读取图片
        img = Image.open(A_img_paths + train_x_name)
        img = img.resize(size)
        img_array = np.array(img)

        img_array = img_array / 255.0  # 标准化
        img_array = img_array * 2 - 1
        x_train.append(img_array)

        # 2. 获取训练样本标签的名字
        train_y_name = lines[read_line].split(',')[1].replace('\n', '')

        train_teacher_y_name = lines[read_line].split(',')[0].replace('\n', '')[:-4] + '.jpg'

        # 根据图片名字读取图片
        img_array = cv2.imread(B_img_paths + train_y_name)
        if img_array.shape == (600, 800, 3):
            img_array = cv2.dilate(img_array, kernel=(5, 5), iterations=5)
        img_array = cv2.dilate(img_array, kernel=(3, 3), iterations=3)
        img_teacher_array = cv2.imread(C_img_paths + train_teacher_y_name, cv2.IMREAD_GRAYSCALE)

        # img.show()
        # print(train_y_name)
        # img = img.resize(size)  # 改变图片大小 -> (227, 227)
        # img_array = np.array(img)
        # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

        # 生成标签， 标签的shape是（size[0]， size[1]， class_numbers) = (227, 227, 2), 里面的值全是0
        labels = np.zeros((size[0], size[1], 2), np.int)

        # 下面将(224,224,3) => (224,224,2),不仅是通道数的变化，还有，
        # 原本背景和裂缝在一个通道里面，现在将斑马线和背景放在不同的通道里面。
        # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
        # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
        # 相当于合并的图层分层！！！！

        labels[:, :, 0] = (img_array[:, :, 0] == 255).astype(int).reshape(size)
        labels[:, :, 0] = (img_array[:, :, 0] == 255).astype(int).reshape(size)
        labels[:, :, 1] = (img_array[:, :, 0] != 255).astype(int).reshape(size)

        # teacher_label = ((img_teacher_array - 127.5) / 127.5).astype(np.float32).reshape(512, 512, 1)
        # teacher_label_opposite = 1 - teacher_label
        # teacher_label = np.concatenate([teacher_label, teacher_label_opposite], axis=2)

        y_train.append(labels)
        # y_teacher_train.append(teacher_label)

    if not KD:
        return np.array(x_train), np.array(y_train)

    if KD:
        return np.array(x_train), {'Output_Label': np.array(y_train), 'Soft_Label': np.array(y_teacher_train)}


# 开始构建自己的标准化的Data_Loader
# 可以获取文件路径，然后再使用map函数进行transform
def load_data_from_filelist(filelist, batch_size, buffer_size, map_func):
    data_filelist = tf.constant(filelist)
    dataset = tf.data.Dataset.from_tensor_slices(data_filelist)

    batch_size = batch_size
    # 对数据进行map，完成数据路径 Str 向 Data 的transform
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 对数据进行Batch，Shuffle，Repeat操作
    dataset = dataset.batch(batch_size=batch_size).shuffle(buffer_size=buffer_size).repeat()

    data_iter = iter(dataset)
    while True:
        try:
            data = next(data_iter)
            print(data[0].shape)
        except StopIteration:
            print('Load End')


# 设计常用的map函数
def map_function(file_path: str or list,
                 file_mode: str,
                 label_mode: str or None or int,
                 label_num: int or None):
    """
    无标签的图像Load_Map函数
    带数字标签的图像Load_Map函数
    :param label_num: 标签的种类数目
    :param label_mode: 标签的格式
    :param file_mode: 指定图像的格式
    :param file_path: 图像数据的绝对路径
    :return: 单张图像数据的Tensor
    """

    def data_load(img_data_file_path):

        data_byte = tf.io.read_file(img_data_file_path)
        if file_mode == 'png':
            data_decode = tf.image.decode_png(data_byte)
        elif file_mode == 'jpg':
            data_decode = tf.image.decode_jpeg(data_byte)
        # 设置Tensor数据的数据形式
        data_decode = tf.cast(data_decode, tf.float32)
        data_decode = (data_decode - 127.5) / 255.

        return data_decode

    def label_load(img_label_file_path):
        # 假设是图像分类的问题，Label是数字，即img_label_file_path是数字
        label = tf.constant(img_label_file_path)
        label = tf.one_hot(indices=label, depth=label_num, on_value=1., off_value=0., axis=-1)
        # print(label.shape)
        return label

    if file_path is str:
        data = data_load(file_path)
        return data

    if file_path is list:
        data_file_path = file_path[0]
        data = data_load(data_file_path)
        label_file_path = file_path[1]

        if label_mode is int:
            label = label_load(label_file_path)
            return data, label

        elif label_mode is str:
            # 这种情况是指两张对应图片的情况
            label = data_load(label_file_path)
            return data, label


def get_teacher_dataset_label \
                (
                lines,
                A_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\img/',
                B_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\mask/',
                h_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_h\label/',
                x_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_x\label/',
                y_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_y\label/',
                mix_img_paths=r'L:\ALASegmentationNets\Data\Stage_4\train\teacher_mask\teacher_label_mix\label/',
                batch_size=1,
                shuffle=True,
                temperature=0
        ):
    numbers = len(lines)
    read_line = 0

    while True:
        x_train = []
        y_train = []

        for t in range(batch_size):
            if shuffle:
                np.random.shuffle(lines)

            train_x_name = lines[read_line].split(',')[0]

            # 根据图片名字读取图片
            img = cv2.imread(A_img_paths + train_x_name)
            size = (img.shape[0], img.shape[1])

            img_array = img / 255.0  # 标准化
            img_array = img_array * 2 - 1
            x_train.append(img_array)

            # 根据相应标签载入真实标签
            train_y_name = lines[read_line].split(',')[1].replace('\n', '')
            img_array = cv2.imread(B_img_paths + train_y_name)
            img_array = cv2.dilate(img_array, kernel=(3, 3), iterations=3)

            labels = np.zeros((img_array.shape[0], img_array.shape[1], 2), np.int)

            labels[:, :, 0] = (img_array[:, :, 1] == 255).astype(int).reshape(size)
            labels[:, :, 1] = (img_array[:, :, 1] != 255).astype(int).reshape(size)
            real_label = labels.astype(np.float32)
            if temperature > 0:
                real_label = tf.nn.softmax(real_label / temperature)
                real_label = np.array(real_label, dtype=np.float32)

            # 根据相应标签载入相应的老师标签
            def get_label(img_paths):

                label = cv2.imread(img_paths + train_x_name[:-4] + '.png')
                label = label[:, :, 0:1]
                label = label / 255.0

                label_T = 1 - label
                label = np.concatenate([label, label_T], axis=-1)

                label = np.array(label)
                return label

            h_label = get_label(h_img_paths)
            x_label = get_label(x_img_paths)
            y_label = get_label(y_img_paths)
            mix_label = get_label(mix_img_paths)

            y_train.append(h_label)
            y_train.append(x_label)
            y_train.append(y_label)
            y_train.append(mix_label)

            y_train.append(real_label)

            read_line = (read_line + 1) % numbers

        yield np.array(x_train), np.array(y_train)
