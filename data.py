import h5py
import numpy as np
import os


def GetData(data_folder):
    file_list = os.listdir(data_folder)
    image_list = []
    label_list = []

    for file in file_list:
        file_path = os.path.join(data_folder, file)

        # data read
        h5_file = h5py.File(file_path, 'r')
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)
        h5_file.close()

        image_list.append(image)
        label_list.append(label)

    return np.asarray(image_list), np.asarray(label_list)


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







