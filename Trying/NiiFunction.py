import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from MeDIT.SaveAndLoad import LoadNiiData


def Showhist(image_array):
    plt.hist(image_array.flatten(), bins=50)
    plt.show()

# 画roi
def DrawNiiROI(data, roi, roi_save_path):
    for i in range(data.shape[-1]):
        one_image = os.path.join(roi_save_path, str(i) + '.png')
        plt.axis('off')
        plt.imshow(data[:, :, i], cmap='gray')
        plt.contour(roi[:, :, i], colors='g')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(one_image, bbox_inches='tight')
        plt.show()


def DrawCGROI(data_array, CGroi_array, roi_array, cgroi_save_path):
    for i in range(data_array.shape[-1]):
        one_image_path = os.path.join(cgroi_save_path, str(i) + '.png')
        plt.axis('off')
        plt.contour(roi_array[:, :, i], colors='g')
        plt.contour(CGroi_array[:, :, i], colors='r')
        plt.imshow(data_array[:, :, i], cmap='gray')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(one_image_path, bbox_inches='tight')
        plt.show()
    return 0


def DrawROI(data_folder):
    file_list = os.listdir(data_folder)
    for i in range(len(file_list)):
        print(i)
        save_path = os.path.join(data_folder, file_list[i])

        # LoadPath
        data_path = os.path.join(data_folder, file_list[i] + '/t2.nii')
        CGroi_path = os.path.join(data_folder, file_list[i] + '/t2_roi_hy.nii')
        roi_path = os.path.join(data_folder, file_list[i] + '/prostate_roi_TrumpetNet.nii')
        # 如果path不存在则进行下一个循环并报出哪一个路径不存在
        # ############

        # SavePath
        roi_save_path = os.path.join(save_path, 'image0')
        cgroi_save_path = os.path.join(save_path, 'image')

        # LoadData
        _, _, data_array = LoadNiiData(data_path)
        _, _, CGroi_array = LoadNiiData(CGroi_path)
        _, _, roi_array = LoadNiiData(roi_path)

        # DrawROI
        if os.path.exists(roi_save_path):
            DrawNiiROI(data_array, roi_array, roi_save_path)
        else:
            os.mkdir(roi_save_path)
            DrawNiiROI(data_array, roi_array, roi_save_path)

        # DrawCGROI
        if os.path.exists(cgroi_save_path):
            pass
        else:
            os.mkdir(cgroi_save_path)
            DrawCGROI(data_array, CGroi_array, roi_array, cgroi_save_path)

# Get PZ
def GetROI(t2CG_array, t2WG_array):
    PZ = t2WG_array - t2CG_array
    PZ[np.where(PZ == -1)] = 0
    CG = t2CG_array
    CG[np.where(CG == 1)] = 2
    ROI = CG + PZ
    return np.array(PZ), np.array(ROI)



def SavePZ(roi_array, t2_path, store_path):
    roi_array = roi_array.transpose(2, 0, 1)
    t2_image = sitk.ReadImage(t2_path)
    roi_image = sitk.GetImageFromArray(roi_array)
    roi_image.CopyInformation(t2_image)
    sitk.WriteImage(roi_image, store_path)


def GetAndSave(data_folder):
    file_list = os.listdir(data_folder)
    for i in range(len(file_list)):
        t2ROI_save_path = os.path.join(data_folder, file_list[i] + '/t2_CG')
        print(i)

        if os.path.exists(t2ROI_save_path):
            pass
        else:
            os.mkdir(t2ROI_save_path)

        # LoadPath
        t2_path = os.path.join(data_folder, file_list[i] + '/t2.nii')
        t2CG_path = os.path.join(data_folder, file_list[i] + '/t2_roi_hy.nii')
        t2WG_path = os.path.join(data_folder, file_list[i] + '/prostate_roi_TrumpetNet.nii')
        t2ROI_path = os.path.join(t2ROI_save_path, 't2_ROI_roi.nii.gz')
        _, _, t2_array = LoadNiiData(t2_path)
        _, _, t2CG_array = LoadNiiData(t2CG_path)
        _, _, t2WG_array = LoadNiiData(t2WG_path)
        _, t2ROI_array = GetROI(t2CG_array, t2WG_array)
        SavePZ(t2ROI_array, t2_path, t2ROI_path)


data_folder = r'H:/data/TZ roi/data/Validation'
filepath = r'H:/data/TZ roi/data/Train/CAI YUE TANG/image'
savepath = r'C:/Users/I/Desktop'

from MeDIT.Visualization import Imshow3DArray
t2path = r'H:/data/TZ roi/data/Validation/Chen shao qun/t2.nii'
datapath = r'H:/data/TZ roi/data/Validation/Chen shao qun/t2_resample.nii'
roipath = r'H:/data/TZ roi/data/Validation/Chen shao qun/t2_CG/t2_resample_roi.nii'
t2_image, _, t2 = LoadNiiData(t2path, is_show_info=True)
data_image, _, data = LoadNiiData(datapath, is_show_info=True)
roi_image, _, roi = LoadNiiData(roipath, is_show_info=True)
Imshow3DArray(t2)
