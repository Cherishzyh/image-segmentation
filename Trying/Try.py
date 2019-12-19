from random import shuffle

import h5py
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

# from FilePath.PzFilepath import test_folder

#################################################################################
def ReadH5(path):
    for index in range(len(path)):
        with h5py.File(path, 'r') as h5_file:
            image = np.asarray(h5_file['input_0'], dtype=np.float32)
            label_0 = np.asarray(h5_file['output_0'], dtype=np.uint8)
            label_1 = np.asarray(h5_file['output_1'], dtype=np.uint8)
            label_2 = np.asarray(h5_file['output_2'], dtype=np.uint8)
    return image, label_0, label_1, label_2

################################################################################
# Copy t2 resize nii data
def Copyt2():
    original_folder = r'D:\ZYH_DrawPzRoi\T2_Resize_Data'
    target_folder = r'D:\ZYH_DrawPzRoi\Nii'
    case_list = os.listdir(original_folder)
    for case_name in case_list:

        case_folder = os.path.join(original_folder, case_name)

        target_case_folder = os.path.join(target_folder, case_name)
        cg_path = os.path.join(case_folder, 'SeriesRois\\' + 't2_Resize.nii_CG.nii')
        pz_path = os.path.join(case_folder, 'SeriesRois\\' + 't2_Resize.nii_PZ.nii')
        t2_path = os.path.join(case_folder, 't2_Resize.nii')
        # print(cg_path)

        if os.path.exists(cg_path):
            print(case_name)
            if not os.path.exists(target_case_folder):
                os.mkdir(target_case_folder)
            shutil.copy(t2_path, target_case_folder)
            shutil.copy(pz_path, target_case_folder)
            shutil.copy(cg_path, target_case_folder)

#################################################################################
def CheckCgPz():
    path = r'D:\ZYH_DrawPzRoi\T2_Resize_Data'
    file_list = os.listdir(path)
    number = 0
    for file in file_list:
        file_path = os.path.join(path, file + '\SeriesRois')
        cg_path = os.path.join(file_path, 't2_Resize.nii_CG.nii')
        pz_path = os.path.join(file_path, 't2_Resize.nii_PZ.nii')
        if os.path.exists(cg_path) or os.path.exists(pz_path):
            number += 1
            print(file)

    print(number)


#################################################################################
def CopyProstateX():
    from MIP4AIM.Application2Series.SeriesMatcher import SeriesStringMatcher
    t2_matcher = SeriesStringMatcher(include_key='t2_tse_tra', exclude_key=['roi', 'ROI', 'diff', 'ca'])
    dwi_matcher = SeriesStringMatcher(include_key='diff', exclude_key=['Reg', 'ADC', 'BVAL', 'copy'], suffex=('.nii', '.bval', '.bvec', ))
    adc_matcher = SeriesStringMatcher(include_key='ADC', exclude_key=['Reg'], suffex=('.nii', ))
    roi_matcher = SeriesStringMatcher(include_key=['ROI', 'CA'], exclude_key=['XX'], suffex=('.nii', ))

    import os
    file_folder = r'Z:\PrcoessedData\ProcessSongYang'
    copy_folder = r'Z:\PrcoessedData\CopyProstate'
    case_list = os.listdir(file_folder)
    for case_num in range(len(case_list)):

        case_folder = os.path.join(file_folder, case_list[case_num])
        series_list = os.listdir(case_folder)

        roi_result = roi_matcher.Match(series_list)

        if len(roi_result) == 0:
            continue

        t2_result = t2_matcher.Match(series_list)
        dwi_result = dwi_matcher.Match(series_list)
        adc_result = adc_matcher.Match(series_list)

        # print(roi_result)
        for i in range(len(roi_result)):
            index = roi_result[i].index('.')
            copy_name = roi_result[i][19:21] + '_' + roi_result[i][25:index]
            case_name = case_list[case_num][0:8] + '_' + case_list[case_num][10:]
            copy_case_folder = os.path.join(copy_folder, copy_name + '_' + case_name)
            os.mkdir(copy_case_folder)
            # print(copy_case_folder)

            t2_path = os.path.join(case_folder, t2_result[0])
            dwi_path_1 = os.path.join(case_folder, dwi_result[0])
            dwi_path_2 = os.path.join(case_folder, dwi_result[1])
            dwi_path_3 = os.path.join(case_folder, dwi_result[2])
            adc_path = os.path.join(case_folder, adc_result[0])
            roi_path = os.path.join(case_folder, roi_result[i])

            print(case_list[case_num])
            print(copy_case_folder)

            shutil.copy(t2_path, copy_case_folder)
            shutil.copy(dwi_path_1, copy_case_folder)
            shutil.copy(dwi_path_2, copy_case_folder)
            shutil.copy(dwi_path_3, copy_case_folder)
            shutil.copy(adc_path, copy_case_folder)
            shutil.copy(roi_path, copy_case_folder)


#################################################################################
def CheckData():
    import matplotlib.pyplot as plt
    from MeDIT.Normalize import Normalize01

    # case_folder = r'D:\ZYH\Data\TZ roi\Input_0_Output_0\TzRoiAdd65\train'
    case_folder = r'D:\ZYH\Data\TZ roi\Input_0_output_downsample_3\FormatH5\test'
    case_list = os.listdir(case_folder)
    for i in range(len(case_list)):
        case_path = os.path.join(case_folder, case_list[i])
        with h5py.File(case_path, 'r') as h5_file:
            image = np.asarray(h5_file['input_0'], dtype=np.float32)
            label0 = np.asarray(h5_file['output_0'], dtype=np.uint8)
            label1 = np.asarray(h5_file['output_1'], dtype=np.float32)
            label2 = np.asarray(h5_file['output_2'], dtype=np.uint8)
            plt.subplot(221)
            plt.imshow(Normalize01(image), cmap='gray')
            plt.title('{:.3f}-{:.3f}'.format(image.max(), image.min()))
            plt.subplot(222)
            plt.imshow((np.argmax(label0, axis=-1)), cmap='gray')
            plt.title('{}-{}, size:{}'.format(label0.max(), label0.min(), label0.shape))
            plt.subplot(223)
            plt.imshow((np.argmax(label1, axis=-1)), cmap='gray')
            plt.title('{}-{}, size:{}'.format(label1.max(), label1.min(), label1.shape))
            plt.subplot(224)
            plt.imshow((np.argmax(label2, axis=-1)), cmap='gray')
            plt.title('{}-{}, size:{}'.format(label2.max(), label2.min(), label2.shape))
            plt.show()


def ModelPre():
    import numpy as np

    from CNNModel.SuccessfulModel.ProstateSegment import ProstateSegmentationTrumpetNet
    # from MeDIT.UsualUse import *
    from MeDIT.SaveAndLoad import LoadNiiData
    from MeDIT.Normalize import Normalize01
    from MeDIT.Visualization import Imshow3DArray

    model_path = r'd:\SuccessfulModel\ProstateSegmentTrumpetNet'

    segment = ProstateSegmentationTrumpetNet()
    if not segment.LoadConfigAndModel(model_path):
        print('Load Failed')

    t2_image, _, t2 = LoadNiiData(r'd:\Data\HouYing\processed\BIAN JIN YOU\t2.nii')

    preds, mask, mask_image = segment.Run(t2_image)

    show_data = np.concatenate((Normalize01(t2), np.clip(preds, a_min=0.0, a_max=1.0)), axis=1)
    show_roi = np.concatenate((mask, mask), axis=1)

    Imshow3DArray(show_data, roi=show_roi)


def ShowData():
    from MeDIT.SaveAndLoad import LoadNiiData
    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    import matplotlib.pyplot as plt

    folder = r'Z:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii'
    case_list = os.listdir(folder)
    for case in case_list:
        print(case)
        case_path = os.path.join(folder, case)
        t2_path = os.path.join(case_path, 't2_Resize.nii')
        pz_path = os.path.join(case_path, 'ROI_jkw0.nii')
        cg_path = os.path.join(case_path, 'ROI_jkw1.nii')
        if not os.path.exists(cg_path):
            print('cg not exists')
            continue
        if not os.path.exists(pz_path):
            print('pz not exists')
            continue

        _, _, t2_data = LoadNiiData(t2_path, dtype=np.float32, is_show_info=False)
        _, _, pz_data = LoadNiiData(pz_path, dtype=np.float32, is_show_info=False)
        _, _, cg_data = LoadNiiData(cg_path, dtype=np.float32, is_show_info=False)

        Imshow3DArray(Normalize01(t2_data), roi=[pz_data, cg_data])

##########################################################################


if __name__ == '__main__':
    ShowData()




