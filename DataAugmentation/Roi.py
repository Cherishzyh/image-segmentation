import SimpleITK as sitk
import numpy as np
import os
from MeDIT.SaveAndLoad import LoadNiiData


def GetPZ(t2CG_array, t2WG_array):
    import matplotlib.pyplot as plt
    PZ = t2WG_array - t2CG_array
    PZ[np.where(PZ == -1)] = 0
    CG = t2CG_array
    CG[np.where(CG == 1)] = 2
    ROI = CG + PZ
    return np.array(ROI)


def main():
    # LoadPath
    data_folder = r'H:/data/TZ roi/data/Train'
    file_list = os.listdir(data_folder)
    t2PZ_save_path = r'H:/data/TZ roi/data/Train/t2_CG'
    t2_path = os.path.join(data_folder, file_list[25] + '/t2.nii')
    t2CG_path = os.path.join(data_folder, file_list[25] + '/t2_roi_hy.nii')
    t2WG_path = os.path.join(data_folder, file_list[25] + '/prostate_roi_TrumpetNet.nii')
    t2PZ_path = os.path.join(t2PZ_save_path, 't2_PZ_roi.nii.gz')
    t2_image, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
    t2CG_image, _, t2CG_roi = LoadNiiData(t2CG_path, dtype=np.uint8)
    t2WG_image, _, t2WG_roi = LoadNiiData(t2WG_path, dtype=np.uint8)
    Roi_array = GetPZ(t2CG_roi, t2WG_roi)


if __name__ == '__main__':
    main()
