import os
import shutil

folder_path = r'H:\SKMR-ImageProcess'

# 遍历文件夹内的东西
# for parent_path, folder_name, file_name in os.walk(folder_path):
#     print(parent_path)
#     print(folder_name)
#     print(file_name)
#     break

# 复制文件
file_path = r'H:\SKMR-ImageProcess\ProstateX\data1.nii.gz'
target_path = r'H:\SKMR-ImageProcess\demol.nii.gz'
# 必须接文件名
shutil.copy(file_path, target_path)

# 创建文件夹
# os.mkdir()

