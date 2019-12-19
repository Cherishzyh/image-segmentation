import h5py
import numpy as np
import os
from MeDIT.SaveAndLoad import LoadNiiData
import SimpleITK as sitk
import matplotlib.pyplot as plt
# from NiiProcess.Resampler import Resampler


def ReadH5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)
    return image, label


def writeh5(data_folder, save_path):
    file_list = os.listdir(data_folder)
    number = 0
    for i in range(len(file_list)):
        # path
        data_path = os.path.join(data_folder, file_list[i])
        t2_path = os.path.join(data_path, 't2_resample.nii')
        roi_path = os.path.join(data_path, 't2_CG/t2_resample_roi.nii')

        # Load data
        _, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
        _, _, roi = LoadNiiData(roi_path, dtype=np.uint8)

        for j in range(t2.shape[-1]):
            number += 1
            t2_slice = t2[..., j]
            roi_slice = roi[..., j]

            dataname = 'data'+ str(number) + '.h5'
            datapath = os.path.join(save_path, dataname)

            f = h5py.File(datapath, 'w')
            f['input_0'] = t2_slice
            f['output_0'] = roi_slice
            f.close()
            print(number)


def showh5data(filepath):
    file_list = os.listdir(filepath)
    for i in range(len(file_list)):
        path = os.path.join(filepath, file_list[i])
        h5_file = h5py.File(path, 'r')
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)
        # plt.imshow(image[:, :, 0], cmap='gray')
        # plt.show()
        plt.imshow(label[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.show()
        # h5_file.close()


def ReResolution(data_folder):
    file_list = os.listdir(data_folder)
    for i in range(len(file_list)):
        # LoadPath
        t2_path = os.path.join(data_folder, file_list[i] + '/t2.nii')
        roi_path = os.path.join(data_folder, file_list[i] + '/t2_CG/t2_ROI_roi.nii.gz')
        t2_resample_path = os.path.join(data_folder, file_list[i] + '/t2_resample.nii')
        roi_resample_path = os.path.join(data_folder, file_list[i] + '/t2_CG/t2_resample_roi.nii.gz')
        t2_image, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
        roi_image, _, roi = LoadNiiData(roi_path, dtype=np.uint8)
        resampler = Resampler()
        resampler.ResizeSipmleITKImage(t2_image, expected_resolution=[0.5, 0.5, -1], store_path=t2_resample_path)
        resampler.ResizeSipmleITKImage(roi_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                       store_path=roi_resample_path)
        print(i)


#######################################################


def TestShowHist(dataarray, title):
    plt.hist(dataarray.flatten(), bins=50)
    plt.title(title)
    plt.show()
    return 0


def TestShowRoiHist(data_folder):
    file_list = os.listdir(data_folder)
    for i in range(len(file_list)):
        # path
        data_path = os.path.join(data_folder, file_list[i])
        roi_path = os.path.join(data_path, 't2_CG/t2_ROI_roi.nii')
        roi_resample_path = os.path.join(data_path, 't2_CG/t2_resample_roi_try.nii')

        # Load data
        roi_image, _, roi = LoadNiiData(roi_path, dtype=np.float32)
        roi_resample_image, _, roi_resample = LoadNiiData(roi_resample_path, dtype=np.uint8)

        # ShowHist(roi, 'roi')
        # ShowHist(roi_resample, 'roi_resample_try')


def TestShowRoiImage(data_folder):
    file_list = os.listdir(data_folder)
    for i in range(len(file_list)):
        print(i)
        # path
        data_path = os.path.join(data_folder, file_list[i])
        roi_path = os.path.join(data_path, 't2_CG/t2_ROI_roi.nii')
        roi_resample_path = os.path.join(data_path, 't2_CG/t2_resample_roi_try.nii')

        # Load data
        roi_image, _, roi = LoadNiiData(roi_path, dtype=np.uint8)
        roi_resample_image, _, roi_resample = LoadNiiData(roi_resample_path, dtype=np.uint8)

        plt.subplot(121)
        plt.imshow(roi_image, cmap='gray')

        plt.subplot(122)
        plt.imshow(roi_resample_image, cmap='gray')

        plt.axis('off')
        plt.show()


