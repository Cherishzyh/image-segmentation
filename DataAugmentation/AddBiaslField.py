import os
import numpy as np
from matplotlib import pyplot as plt
from MeDIT.Normalize import Normalize01


def BiasField(input_shape, center, drop_ratio):
    parameters = []
    center_x, center_y = center[0], center[1]
    max_x = max(input_shape[0]-center_x, center_x)
    max_y = max(input_shape[1]-center_y, center_y)

    parameters.append(-(drop_ratio) / (2*max_x*max_x))
    parameters.append(-(drop_ratio + parameters[0]*max_x*max_x) / (max_y*max_y))

    field = np.zeros(shape=input_shape)
    for row in range(field.shape[0]):
        for column in range(field.shape[1]):
            field[row][column] = parameters[0]*(row - center_x)*(row - center_x) \
                                 + parameters[1]*(column - center_y)*(column - center_y) + 1
    return field


def AddBiasField(image, drop_ratio, center=[]):
    shape = image.shape
    field = BiasField(shape, center, drop_ratio)
    min_value = image.min()
    # new_image = np.multiply(field, image - min_value) + min_value

    new_image = field*Normalize01(image)
    return new_image, field


def Normalize_11(data):
    new_data = np.asarray(data, dtype=np.float32)
    new_data = 2*(data - np.min(data))/(np.max(data) - np.min(data)) - 1
    return new_data


if __name__ == '__main__':
    from DataProcess.Data import GetData
    import random
    data_folder = r'D:\ZYH\Data\TZ roi\Input_0_Output_0\TzRoiAdd65\test'
    image, _, name = GetData(data_folder)

    for index in range(len(image)):
        center = [random.randint(0,300), random.randint(0,300)]
        drop_ratio = 0.9999
        new_image, field = AddBiasField(image[index], drop_ratio, center)

        plt.figure(figsize=(16, 10))
        plt.subplot(131)
        plt.title('add_bias')
        plt.imshow(new_image, cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.title('original_image')
        plt.imshow(image[index], cmap='gray')
        plt.axis('off')
        plt.subplot(133)
        plt.title('bias' + str(center))
        plt.imshow(field, cmap='gray')
        plt.axis('off')
        plt.show()




