import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# data_path = r'H:/t2/t2.nii'

# 读取图像
# image = sitk.ReadImage(data_path)


# 获得图像信息
def ImageInformation(image):
    image_GetSpacing = image.GetSpacing()
    image_GetDrection = image.GetDrection()
    image_origin = image.GetOrigin()
    print('GetSpacing:', image_GetSpacing)
    print('Size:', image_GetDrection)
    print('Origin:', image_origin)


# 获取图像矩阵信息
def ArrayInformation(image):
    image_array = sitk.GetArrayFromImage(image)
    print(image_array)


# 转换矩阵方向
# transfored_image_array = image_array.transpose((1, 2, 0))

# 从图像中获得矩阵，并Copy原图像的信息
# transfored_image = sitk.GetImageFromArray(transfored_image_array)
# transfored_image.CopyInformation(image)

# 图像保存
# sitk.WriteImage(transfored_image, r'vvv')


# 图像显示
# show_slice = transfored_image_array[:, :, 12]
# plt.imshow(show_slice, cmap='gray')
# plt.show()

# 画像素直方图
# plt.hist(transfored_image_array.flatten(), bins=50)
# plt.show()

# print(data_array.shape)
# print(roi_array.shape)
# max_slice_index = np.argmax(np.sum(roi_array, (0, 1)))
# print(max_slice_index)
# ShowHist(roi_array[:, :, max_slice_index])
# 显示roi内部像素直方图
# ShowHist(data_array[roi_array == 1])
# print(data_array.shape(-1))

def ReDrawRoi(t2_file_path, t2label_file_path, is_show_info=False):
    t2image = sitk.ReadImage(t2_file_path)

    resolution = t2image.GetSpacing()
    direction = t2image.GetDirection()
    origin = t2image.GetOrigin()

    Roiimage = sitk.ReadImage(t2label_file_path)
    Roiimage.SetSpacing(resolution)
    Roiimage.SetDirection(direction)
    Roiimage.SetOrigin(origin)

    if is_show_info:
        print('t2 image information')
        print('Image size is: ', t2image.GetSize())
        print('Image resolution is: ', t2image.GetSpacing())
        print('Image direction is: ', t2image.GetDirection())
        print('Image Origion is: ', t2image.GetOrigin())
        print('\t')
        print('Roi image information')
        print('Image size is: ', Roiimage.GetSize())
        print('Image resolution is: ', Roiimage.GetSpacing())
        print('Image direction is: ', Roiimage.GetDirection())
        print('Image Origion is: ', Roiimage.GetOrigin())

    return t2image, Roiimage


def ReSaveROI(save_store):
    file_list = []
    file_list = os.listdir(save_store)
    for i in range(len(file_list)):
        data_path = os.path.join(save_store, file_list[i])
        data_list = os.listdir(data_path)
        t2_file_path = os.path.join(data_path, data_list[1])
        t2label_file_path = os.path.join(data_path, data_list[0])
        save_roi_path = os.path.join(data_path,  't2_re_label.nii')
        image, Roi = ReDrawRoi(t2_file_path, t2label_file_path)
        sitk.WriteImage(Roi, save_roi_path)
    return 0

save_store = r'H:/data/TZ roi1'
