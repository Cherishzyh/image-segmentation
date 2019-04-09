import matplotlib.pyplot as plt
import h5py
import numpy as np
import os


def GeneratorData(data_folder, batch_size):
    file_list = os.listdir(data_folder)
    image_list = []
    label_list = []

    while True:
        for file in file_list:
            file_path = os.path.join(data_folder, file)

            # data read
            h5_file = h5py.File(file_path, 'r')
            image = np.asarray(h5_file['input_0'], dtype=np.float32)
            label = np.asarray(h5_file['output_0'], dtype=np.uint8)
            h5_file.close()

            image_list.append(image)
            label_list.append(label)

            if len(image_list) >= batch_size:
                yield np.asarray(image_list), np.asarray(label_list)
                image_list = []
                label_list = []


def test_GeneratorData():
    data_folder = r'H:/data/Input_1_Output_1/testing'
    batch_size = 12
    for image_list, label_list in GeneratorData(data_folder, batch_size):
        for i in range(batch_size):
            plt.subplot(batch_size/4, 4, i+1)
            plt.imshow(image_list[i, :, :, 0], cmap='gray')
        plt.show()


# test_GeneratorData()






