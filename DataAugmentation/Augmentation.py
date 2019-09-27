import h5py
import os

import numpy as np
from keras.utils import to_categorical

from MeDIT.DataAugmentor import DataAugmentor2D, AugmentParametersGenerator



'''数据预处理流程：.nii→改分辨率→保存为.h5→进行数据增强，图片剪裁，onehot编码→generator'''

def AugmentTrain(train_folder, batch_size):
    file_list = os.listdir(train_folder)
    image_list = []
    label_list = []
    for i in range(len(file_list)):
        # path
        data_path = os.path.join(train_folder, file_list[i])
        h5_file = h5py.File(data_path, 'r')
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)

        # augmentation param
        param_dict = {'stretch_x': 0.1, 'stretch_y': 0.1, 'shear': 0.1, 'rotate_z_angle': 20, 'horizontal_flip': True}

        # 设增强函数为augment_generator
        augment_generator = AugmentParametersGenerator()

        # 2D的数据增强
        augmentor = DataAugmentor2D()
        augment_generator.RandomParameters(param_dict)
        transform_param = augment_generator.GetRandomParametersDict()
        augment_t2 = augmentor.Execute(image, aug_parameter=transform_param, interpolation_method='linear')
        augment_roi = augmentor.Execute(label, aug_parameter=transform_param, interpolation_method='linear')

        # cut
        if np.shape(augment_t2) == (440, 440):
            cropImage = image[100:340, 100:340]
            cropRoi = label[100:340, 100:340]
            cropaugmentImage = augment_t2[100:340, 100:340]
            cropaugmentRoi = augment_roi[100:340, 100:340]

        else:
            cropImage = image[60:300, 60:300]
            cropRoi = label[60:300, 60:300]
            cropaugmentImage = augment_t2[60:300, 60:300]
            cropaugmentRoi = augment_roi[60:300, 60:300]

        # one_hot
        roi_onehot = to_categorical(cropRoi)
        augment_roi_onehot = to_categorical(cropaugmentRoi)

        # show
        # plt.imshow(np.concatenate((Normalize01(cropImage), Normalize01(cropRoi)), axis=1), cmap='gray')
        # plt.show()
        # plt.imshow(np.concatenate((Normalize01(cropaugmentImage), Normalize01(cropaugmentRoi)), axis=1), cmap='gray')
        # plt.show()

        reshape = (240, 240, 1)
        cropImage = cropImage.reshape(reshape)
        cropaugmentImage = cropaugmentImage.reshape(reshape)

        # add data into list
        image_list.append(cropImage)
        label_list.append(roi_onehot)

        image_list.append(cropaugmentImage)
        label_list.append(augment_roi_onehot)

        if len(image_list) >= batch_size:
            yield np.asarray(image_list), np.asarray(label_list)
            image_list = []
            label_list = []


# path = r'H:\data\data\validation'
# AugmentTrain(path, batch_size=16)


