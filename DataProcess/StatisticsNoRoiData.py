import os
import numpy as np
import matplotlib.pyplot as plt
import h5py


def CopyData(label_list, case_list):
    sum = 0

    for case_num in range(len(case_list)):
        for index in range(label_list[case_num].shape[-1]):
            plt.imshow(label_list[case_num, :, :, index], cmap='gray')
            plt.show()
            # if np.all(label_list[case_num, :, :, index] == 0) or np.all(label_list[case_num, :, :, index] == 1):
        if np.all(label_list[case_num, :, :, 0] == 1):
                print(case_list[case_num])
                # shutil.copy(os.path.join(train_folder, case_list[case_num]),
                #             os.path.join(r'D:\ZYH\Data\TZ roi\TypeOfData\NewFormatH5\validation_nolabel', case_list[case_num]))
                # if index == 2:
                sum += 1
            # else:
                # shutil.copy(os.path.join(train_folder, case_list[case_num]),
                #             os.path.join(r'D:\ZYH\Data\TZ roi\TypeOfData\NewFormatH5\validation', case_list[case_num]))

    print('There are {} image, {} of them are no label.'.format(len(case_list), sum))


def GetH5(file_path):
    # data read
    with h5py.File(file_path, 'r') as h5_file:
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)
    return image, label


def main():

    # _, label_list, case_list = GetData(validation_folder)

    # CopyData(label_list, case_list)
    case_list = os.listdir(r'D:\ZYH\Data\TZ roi\TypeOfData\NewFormatH5\train')
    for index in range(len(case_list)):
        path = os.path.join(r'D:\ZYH\Data\TZ roi\TypeOfData\NewFormatH5\train', case_list[index])
        _, label = GetH5(path)
        for channel in range(3):
            if np.all(label[:, :, channel]) == 0:
                plt.imshow(label[:, :, channel], cmap='gray')
                plt.show()


if __name__ == '__main__':
    main()