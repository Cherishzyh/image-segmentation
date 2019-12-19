import numpy as np
import matplotlib.pyplot as plt
from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Visualization import Imshow3DArray

t2_file_path = r'C:\Users\Cherish\Desktop\t2.nii'
# roi_file_path = r'D:\ZYH\Data\ZYH_DrawPzRoi\T2_Resize_Data\2019-CA-formal-CAO JIAN QIANG\SeriesRois\t2_Resize.nii_CG.nii'

_, _, t2_data = LoadNiiData(t2_file_path, dtype=np.float32)
# _, _, roi_data = LoadNiiData(roi_file_path, dtype=np.float32)
for i in range(t2_data.shape[2]):
    plt.imshow(t2_data[:, :, i], cmap='gray')
    plt.show()
