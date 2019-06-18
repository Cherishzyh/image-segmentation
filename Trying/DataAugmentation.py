from NiiFunction import LoadNiiData,GetPZ
import matplotlib.pyplot as plt
import SimpleITK as sitk

t2_array = LoadNiiData(r'H:/data/TZ roi/BIAN ZHONG BEN/t2.nii')
t2WG_array = LoadNiiData(r'H:/data/TZ roi/BIAN ZHONG BEN/t2_roi_hy.nii')
t2CG_array = LoadNiiData(r'H:/data/TZ roi/BIAN ZHONG BEN/prostate_roi_TrumpetNet.nii')
t2PZ_array = GetPZ(t2CG_array, t2WG_array)
t2_image = sitk.GetImageFromArray(t2_array)

for i in range(t2WG_array.shape[-1]):
    plt.axis('off')
    plt.imshow(t2_array[:, :, i], cmap='gray')
    plt.show()
# def DataAugmention(image):