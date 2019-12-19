import os
import h5py

from random import shuffle
import numpy as np

from MeDIT.DataAugmentor import AugmentParametersGenerator, DataAugmentor2D, random_2d_augment
from CNNModel.Training.Generate import _AddOneSample, _AugmentDataList2D, _GetInputOutputNumber, _MakeKerasFormat
from CNNModel.Utility.BaseProcess import ImageProcess2D


def ImageInImageOut2D(root_folder, input_shape, batch_size=8, augment_param={}):
    input_number, output_number = _GetInputOutputNumber(root_folder)
    case_list = os.listdir(root_folder)

    input_list = [[] for index in range(input_number)]
    output_list = [[] for index in range(output_number)]

    param_generator = AugmentParametersGenerator()
    augmentor = DataAugmentor2D()
    crop = ImageProcess2D()

    while True:
        shuffle(case_list)
        for case in case_list:
            case_path = os.path.join(root_folder, case)
            if not case_path.endswith('.h5'):
                continue

            input_data_list, output_data_list = [], []
            try:
                file = h5py.File(case_path, 'r')
                for input_number_index in range(input_number):
                    temp_data = np.asarray(file['input_' + str(input_number_index)])
                    if temp_data.ndim == 2:
                        temp_data = temp_data[..., np.newaxis]
                    input_data_list.append(temp_data)
                for output_number_index in range(output_number):
                    temp_data = np.asarray(file['output_' + str(output_number_index)])
                    if temp_data.ndim == 2:
                        temp_data = temp_data[..., np.newaxis]
                    output_data_list.append(temp_data)
                file.close()
            except Exception as e:
                print(case_path)
                print(e.__str__())
                continue

            param_generator.RandomParameters(augment_param)
            augmentor.SetParameter(param_generator.GetRandomParametersDict())

            input_data_list = _AugmentDataList2D(input_data_list, augmentor)
            output_data_list = _AugmentDataList2D(output_data_list, augmentor)

            input_data_list = crop.CropDataList2D(input_data_list, input_shape)
            output_data_list = crop.CropDataList2D(output_data_list, input_shape)

            _AddOneSample(input_list, input_data_list)
            _AddOneSample(output_list, output_data_list)

            if len(input_list[0]) >= batch_size:
                inputs = _MakeKerasFormat(input_list)
                outputs = _MakeKerasFormat(output_list)
                return inputs, outputs
                # yield inputs, outputs
                # # input_list = [[] for index in range(input_number)]
                # # output_list = [[] for index in range(output_number)]


if __name__ == '__main__':
    train_folder = r'D:\ZYH\Data\TZ roi\Input_0_output_downsample_3\FormatH5\validation'
    image_shape = (320, 320)
    batch_size = 12

    train_generator = ImageInImageOut2D(train_folder, image_shape, batch_size=batch_size, augment_param=random_2d_augment)
