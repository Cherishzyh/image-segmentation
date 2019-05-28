from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Visualization import Imshow3DArray, DrawBoundaryOfBinaryMask
from MeDIT.Normalize import Normalize01
import numpy as np
from MeDIT.DataAugmentor import DataAugmentor2D, AugmentParametersGenerator
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import argmax
from keras.utils import to_categorical

def AugmentTrain(train_folder, batch_size):
    file_list = os.listdir(train_folder)
    image_list = []
    label_list = []
    for i in range(len(file_list)):
        # path
        data_path = os.path.join(train_folder, file_list[i])
        t2_path = os.path.join(data_path, 't2.nii')
        roi_path = os.path.join(data_path, 't2_CG/t2_PZ_roi.nii.gz')

        # Load data
        t2_image, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
        roi_image, _, roi = LoadNiiData(roi_path, dtype=np.uint8)

        # augmentation param
        param_dict = {'stretch_x': 0.1, 'stretch_y': 0.1, 'shear': 0.1, 'rotate_z_angle': 20, 'horizontal_flip': True}

        # 设增强函数为augment_generator
        augment_generator = AugmentParametersGenerator()

        # 2D的数据增强
        augmentor = DataAugmentor2D()

        for j in range(t2.shape[-1]):
            t2_slice = t2[..., j]
            roi_slice = roi[..., j]
            roi_onehot_slice = to_categorical(roi_slice)
            image_list.append(t2_slice)
            label_list.append(roi_onehot_slice)
            augment_generator.RandomParameters(param_dict)
            transform_param = augment_generator.GetRandomParametersDict()
            augment_t2 = augmentor.Execute(t2_slice, aug_parameter=transform_param, interpolation_method='linear')
            augment_roi = augmentor.Execute(roi_slice, aug_parameter=transform_param, interpolation_method='linear')
            augment_onehot_roi = to_categorical(augment_roi)
            # plt.imshow(np.concatenate((Normalize01(augment_t2), Normalize01(augment_roi)), axis=1), cmap='gray')
            # plt.show()

            # add data into list
            image_list.append(augment_t2)
            label_list.append(augment_onehot_roi)

            if len(image_list) >= batch_size:
                yield np.asarray(image_list), np.asarray(label_list)

def AugmentValidation(validation_folder, batch_size):
    file_list = os.listdir(validation_folder)
    image_list = []
    label_list = []
    for i in range(len(file_list)):
        # path
        data_path = os.path.join(validation_folder, file_list[i])
        t2_path = os.path.join(data_path, 't2.nii')
        roi_path = os.path.join(data_path, 't2_CG/t2_PZ_roi.nii.gz')

        # Load data
        t2_image, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
        roi_image, _, roi = LoadNiiData(roi_path, dtype=np.uint8)

        for j in range(t2.shape[-1]):
            t2_slice = t2[..., j]
            roi_slice = roi[..., j]
            roi_onehot_slice = to_categorical(roi_slice)
            image_list.append(t2_slice)
            label_list.append(roi_onehot_slice)

            if len(image_list) >= batch_size:
                yield np.asarray(image_list), np.asarray(label_list)



# data_folder = r'H:/data/TZ roi/data/Train'
# batch_size = 16
# AugmentValidation(data_folder, batch_size)
# 数据还没有裁剪

