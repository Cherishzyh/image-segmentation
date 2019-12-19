import os
import numpy as np
from MeDIT.SaveAndLoad import LoadNiiData, SaveNiiImage
from MeDIT.ImageProcess import GetImageFromArrayByImage
from MeDIT.Others import CopyFile
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def ExtractTargetFile():
    source_root = r'D:\ProblemData\source'
    dest_root = r'D:\ProblemData\Prcoessed'

    for case in os.listdir(source_root):
        case_folder = os.path.join(source_root, case)
        if not os.path.isdir(case_folder):
            continue

        print('Copying {}'.format(case))
        t2_path = os.path.join(case_folder, 't2.nii')
        prostate_roi_path = os.path.join(case_folder, 'prostate_roi_TrumpetNet_check.nii.gz')
        cg_path = os.path.join(case_folder, 't2_roi_hy.nii')

        dest_case_folder = os.path.join(dest_root, case)
        if os.path.exists(prostate_roi_path) and os.path.exists(cg_path):
            if not os.path.exists(dest_case_folder):
                # print(dest_case_folder)
                os.mkdir(dest_case_folder)

        if os.path.exists(t2_path) and os.path.exists(prostate_roi_path) and os.path.exists(cg_path):
            CopyFile(t2_path, os.path.join(dest_case_folder, 't2.nii'))
            CopyFile(prostate_roi_path, os.path.join(dest_case_folder, 'prostate_roi.nii.gz'))
            CopyFile(cg_path, os.path.join(dest_case_folder, 'cg.nii'))


# ExtractTargetFile()

def Normalize():
    root_folder = r'D:\ProblemData\Prcoessed'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Normalizing PZ of {}'.format(case))
        wg_path = os.path.join(case_folder, 'prostate_roi.nii.gz')
        cg_path = os.path.join(case_folder, 'cg.nii')
        if not os.path.exists(wg_path) and not os.path.exists(cg_path):
            continue

        wg_image, _, wg_data = LoadNiiData(wg_path, dtype=np.uint8)
        cg_image, _, cg_data = LoadNiiData(cg_path, dtype=np.uint8)

        print(np.unique(wg_data), np.unique(cg_data))

        if list(np.unique(wg_data)) == [0, 255]:
            wg = np.array(wg_data / 255, dtype=np.uint8)
        else:
            wg = wg_data

        if list(np.unique(cg_data)) == [0, 255]:
            cg = np.array(cg_data / 255, dtype=np.uint8)
        else:
            cg = cg_data

        wg_array = GetImageFromArrayByImage(wg, wg_image)
        cg_array = GetImageFromArrayByImage(cg, wg_image)
        SaveNiiImage(os.path.join(case_folder, 'normalize_prostate_roi.nii'), wg_array)
        SaveNiiImage(os.path.join(case_folder, 'normalize_cg.nii'), cg_array)
        print(np.unique(wg), np.unique(cg))

# Normalize()


def TestNormalize():
    # cg
    cg_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\ROI_jkw0.nii'
    # wg
    wg_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\ROI_jkw1.nii'
    t2_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\t2_Resize.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path)
    _, _, cg = LoadNiiData(cg_path, dtype=np.uint8)
    _, _, wg = LoadNiiData(wg_path, dtype=np.uint8)


    Imshow3DArray(Normalize01(t2), roi=[Normalize01(cg), Normalize01(wg)])


# TestNormalize()


def GetPzRegion():
    root_folder = r'D:\ProblemData\Prcoessed'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Estimating PZ of {}'.format(case))
        wg_path = os.path.join(case_folder, 'prostate_roi.nii.gz')
        cg_path = os.path.join(case_folder, 'cg.nii')
        t2_path = os.path.join(case_folder, 't2.nii')
        if not os.path.exists(wg_path) and not os.path.exists(cg_path):
            continue

        wg_image, _, wg_data = LoadNiiData(wg_path, dtype=np.uint8)
        cg_image, _, cg_data = LoadNiiData(cg_path, dtype=np.uint8)

        # # Calculating PZ
        pz_data = wg_data - cg_data
        # print(np.unique(pz_data))
        pz_data[pz_data == 255] = 0


        # from MeDIT.Visualization import Imshow3DArray
        # from MeDIT.Normalize import Normalize01
        #
        # _, _, t2 = LoadNiiData(t2_path)
        #
        # Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(t2)], axis=1), roi=np.concatenate([pz_data0, pz_data], axis=1))
        #
        # print(np.unique(pz_data), np.unique(wg_data), np.unique(cg_data))

        # Save PZ image
        pz_image = GetImageFromArrayByImage(pz_data, wg_image)

        SaveNiiImage(os.path.join(case_folder, 'pz.nii'), pz_image)


# GetPzRegion()

def TestGetPzRegion():
    # pz_path = r'X:\PrcoessedData\PzTzSegment_ZYH\DoctorDraw2\2019-CA-formal-JIANG HONG GEN\pz.nii'
    t2_path = r'D:\ProblemData\Prcoessed\Chen xian ming\t2.nii'
    cg_path = r'D:\ProblemData\Prcoessed\Chen xian ming\cg.nii'
    wg_path = r'D:\ProblemData\Prcoessed\Chen xian ming\prostate_roi.nii.gz'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    import matplotlib.pyplot as plt

    _, _, t2 = LoadNiiData(t2_path)
    # _, _, pz = LoadNiiData(pz_path, dtype=np.uint8)
    _, _, cg = LoadNiiData(cg_path, dtype=np.uint8)
    _, _, wg = LoadNiiData(wg_path, dtype=np.uint8)

    Imshow3DArray(Normalize01(t2), roi=[wg, cg])
    # print(np.unique(pz), np.unique(cg), np.unique(wg))


# TestGetPzRegion()


########################################################
def MergePzAndTz():
    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    root_folder = r'D:\ProblemData\Prcoessed'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Estimating PZ of {}'.format(case))
        cg_path = os.path.join(case_folder, 'cg.nii')
        pz_path = os.path.join(case_folder, 'pz.nii')

        cg_image, _, cg_data = LoadNiiData(cg_path, dtype=np.uint8)
        pz_image, _, pz_data = LoadNiiData(pz_path, dtype=np.uint8)

        # Merge
        merge_roi = pz_data + 2 * cg_data
        # Imshow3DArray(np.concatenate((Normalize01(cg_data), Normalize01(pz_data), Normalize01(merge_roi)), axis=1))
        # print(np.unique(merge_roi))

        # Save PZ image
        merge_image = GetImageFromArrayByImage(merge_roi, pz_image)
        SaveNiiImage(os.path.join(case_folder, 'merge_pz1_cg2.nii'), merge_image)


# MergePzAndTz()


def TestMergePzAndTz():
    merge_path = r'Z:\StoreFormatData\PzTzSegment_ZYH\ZYHDraw\2019-CA-formal-BAO TONG\merge_pz1_cg2.nii'
    t2_path = r'Z:\StoreFormatData\PzTzSegment_ZYH\ZYHDraw\2019-CA-formal-BAO TONG\t2.nii'
    pz_path = r'Z:\StoreFormatData\PzTzSegment_ZYH\ZYHDraw\2019-CA-formal-BAO TONG\pz.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path)
    _, _, pz = LoadNiiData(pz_path)
    _, _, merge_roi = LoadNiiData(merge_path, dtype=np.uint8)

    Imshow3DArray(np.concatenate((Normalize01(pz), Normalize01(merge_roi)), axis=1))

# TestMergePzAndTz()

########################################################


def ResampleData():
    from MIP4AIM.NiiProcess.Resampler import Resampler

    root_folder = r'D:\ProblemData\Prcoessed'
    dest_root = r'D:\ProblemData\StoreFormatData'

    resampler = Resampler()
    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        dest_case_folder = os.path.join(dest_root, case)
        if not os.path.exists(dest_case_folder):
            os.mkdir(dest_case_folder)

        print('Resample PZ of {}'.format(case))
        t2_path = os.path.join(case_folder, 't2.nii')
        merge_path = os.path.join(case_folder, 'merge_pz1_cg2.nii')

        t2_image, _, t2_data = LoadNiiData(t2_path)
        merge_image, _, merge_data = LoadNiiData(merge_path, dtype=np.uint8)

        resampler.ResizeSipmleITKImage(t2_image, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 't2_Resize_05x05.nii'))
        resampler.ResizeSipmleITKImage(merge_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 'MergeRoi_Resize_05x05.nii'))


# ResampleData()

def TestResampleData():
    merge_path = r'D:\ProblemData\StoreFormatData\CAI NING_0007947872\MergeRoi_Resize_05x05.nii'
    t2_path = r'D:\ProblemData\StoreFormatData\CAI NING_0007947872\t2_Resize_05x05.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path, is_show_info=True)
    _, _, merge_roi = LoadNiiData(merge_path, dtype=np.uint8, is_show_info=True)

    Imshow3DArray(np.concatenate((Normalize01(t2), Normalize01(merge_roi)), axis=1))


# TestResampleData()


########################################################

def OneHot():
    root_folder = r'D:\ProblemData\StoreFormatData'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print('Onthot coding: {}'.format(case))
        t2_path = os.path.join(case_folder, 't2_Resize_05x05.nii')
        roi_path = os.path.join(case_folder, 'MergeRoi_Resize_05x05.nii')

        _, _, t2 = LoadNiiData(t2_path)
        _, _, merge_roi = LoadNiiData(roi_path, dtype=np.uint8)

        # One Hot
        # 这里3代表PZ, CG, 背景, output : row x column x slices x 3
        output = np.zeros((merge_roi.shape[0], merge_roi.shape[1], merge_roi.shape[2], 3), dtype=np.uint8)
        output[..., 0] = np.asarray(merge_roi == 0, dtype=np.uint8)  # save background
        output[..., 1] = np.asarray(merge_roi == 1, dtype=np.uint8)  # save PZ
        output[..., 2] = np.asarray(merge_roi == 2, dtype=np.uint8)  # save CG

        np.save(os.path.join(case_folder, 't2.npy'), t2)
        np.save(os.path.join(case_folder, 'roi_onehot.npy'), output)


# OneHot()

def TestOneHot():
    t2 = np.load(r'D:\ProblemData\StoreFormatData\CAI NING_0007947872\t2.npy')
    roi_onehot = np.load(r'D:\ProblemData\StoreFormatData\CAI NING_0007947872\roi_onehot.npy')

    print(type(t2), type(roi_onehot))

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    Imshow3DArray(Normalize01(t2), roi_onehot[..., 0])
    Imshow3DArray(Normalize01(t2), roi_onehot[..., 1])
    Imshow3DArray(Normalize01(t2), roi_onehot[..., 2])


# TestOneHot()


#########################################################
def Remove(roi_data):
    import random
    pz_list =[]
    cg_list = []
    ignorance_list = []

    for slice in range(roi_data[..., 1].shape[2]):
        pz_list.append(list(np.unique(roi_data[..., 1][..., slice])))
        cg_list.append(list(np.unique(roi_data[..., 2][..., slice])))

    pz_first = pz_list.index([0, 1])
    pz_last = len(pz_list) - list(reversed(pz_list)).index([0, 1]) - 1
    cg_first = cg_list.index([0, 1])
    cg_last = len(cg_list) - list(reversed(cg_list)).index([0, 1]) - 1

    for ignorance in (pz_first - 1, pz_first - 2, cg_last + 1, cg_last + 2):
        if 0 < ignorance < roi_data[..., 1].shape[2]:
            ignorance_list.append(ignorance)

    for index in range(pz_first, pz_last+1):
        if pz_list[index] == [0]:
            ignorance_list.append(index)

    for index in range(cg_first, cg_last+1):
        if cg_list[index] == [0] and index not in ignorance_list:
            ignorance_list.append(index)
 
    # 0-first
    if pz_first - 2 <= 2:
        pass
    else:
        f = random.sample(range(0, pz_first-3), 2)
        ignorance_list.extend([i for i in range(0, pz_first-2) if i not in f])

    # last - final
    if cg_last + 2 >= len(cg_list) - 2:
        pass
    else:
        f = random.sample(range((cg_last+3), len(cg_list)), 2)
        ignorance_list.extend([i for i in range(cg_last + 3, len(cg_list)) if i not in f])

    return pz_first, cg_last, sorted(ignorance_list)


# Series()
########################################################


def TestRomove():
    root_folder = r'D:\emmmData\Final'

    for case in os.listdir(root_folder):
        case_folder = os.path.join(root_folder, case)
        if not os.path.isdir(case_folder):
            continue

        print(case)

        # t2_data = np.load(os.path.join(case_folder, 't2.npy'))
        roi_data = np.load(os.path.join(case_folder, 'roi_onehot.npy'))
        pz_first, cg_last, ignorance_list = Remove(roi_data)
        print('pz:', pz_first, cg_last, ignorance_list)

# TestRomove()


#########################################################
# Normalization and Crop 2D
def CropT2Data(t2_data, crop_shape, slice_index):
    from MeDIT.ArrayProcess import Crop2DImage
    t2_one_slice = t2_data[..., slice_index]

    # Normalization
    t2_one_slice -= np.mean(t2_one_slice)
    t2_one_slice /= np.std(t2_one_slice)

    # Crop
    t2_crop = Crop2DImage(t2_one_slice, crop_shape[:2])
    return t2_crop


def CropRoiData(roi_data, crop_shape, slice_index):
    from MeDIT.ArrayProcess import Crop3DImage
    roi_one_slice_onehot = roi_data[..., slice_index, :]

    # Crop
    roi_crop = Crop3DImage(roi_one_slice_onehot, crop_shape)
    return roi_crop


def MakeH5():
    from MeDIT.SaveAndLoad import SaveH5
    from MeDIT.Log import CustomerCheck
    from MeDIT.Visualization import LoadWaitBar

    from DataProcess.MaxPool import maxpooling

    source_root = r'D:\ProblemData\StoreFormatData'
    dest_root = r'D:\ProblemData\SuccessfulData'

    crop_shape = (320, 320, 3)  # 如果进入网络是240x240
    my_log = CustomerCheck(r'D:\ProblemData\SuccessfulData\zyh_log.csv')

    for case in os.listdir(source_root):
        case_folder = os.path.join(source_root, case)
        if not os.path.isdir(case_folder):
            continue

        print('Making {} for H5'.format(case))
        t2_data = np.load(os.path.join(case_folder, 't2.npy'))
        roi_data = np.load(os.path.join(case_folder, 'roi_onehot.npy'))

        _, _, ignorance_list = Remove(roi_data)

        new_data = []
        new_roi = []
        for index, slice_index in enumerate(range(t2_data.shape[-1])):
            if index in ignorance_list or slice_index == 0 or slice_index == (t2_data.shape[-1] - 1):
                pass
            else:
                LoadWaitBar(t2_data.shape[-1], index)
                print("\n")

                t2_slice_before_crop = CropT2Data(t2_data, crop_shape, slice_index - 1)
                t2_slice_after_crop = CropT2Data(t2_data, crop_shape, slice_index + 1)
                t2_crop = CropT2Data(t2_data, crop_shape, slice_index)
                roi_crop = CropRoiData(roi_data, crop_shape, slice_index)

                down_sample_once = maxpooling(roi_crop)
                down_sample_twice = maxpooling(down_sample_once)

                new_data.append(t2_crop)
                new_roi.append(roi_crop)

                file_name = os.path.join(dest_root, '{}-slicer_index_{}.h5'.format(case, slice_index))
                SaveH5(file_name, [t2_slice_before_crop, t2_crop, t2_slice_after_crop, roi_crop, down_sample_once, down_sample_twice],
                       ['input_0', 'input_1', 'input_2', 'output_0', 'output_1', 'output_2'])

                my_log.AddOne('{}-slicer_index_{}.h5'.format(case, slice_index), ['', case_folder])
        new_data_array = np.transpose(np.asarray(new_data, dtype=np.float32), (1, 2, 0))
        pz_array = Normalize01(np.transpose(np.asarray(new_roi, dtype=np.uint8)[..., 1], (1, 2, 0)))
        cg_array = Normalize01(np.transpose(np.asarray(new_roi, dtype=np.uint8)[..., 2], (1, 2, 0)))
        Imshow3DArray(Normalize01(new_data_array), roi=[pz_array, cg_array])

# MakeH5()

#########################################################


def Try():
    import h5py
    import matplotlib.pyplot as plt
    case_folder = r'D:\ProblemData\SuccessfulData'
    case_list = os.listdir(case_folder)
    for index, case in enumerate(case_list):
        # print(index, case)
        data_path = os.path.join(case_folder, case)
        with h5py.File(data_path, 'r') as h5_file:
            # print(h5_file.keys())
            image = np.asarray(h5_file['input_1'], dtype=np.float32)
            label = np.asarray(h5_file['output_0'], dtype=np.uint8)
        plt.imshow(image, cmap='gray')
        plt.contour(label[..., 1], color='r')
        plt.title(case)
        plt.axis('off')
        plt.show()


Try()









