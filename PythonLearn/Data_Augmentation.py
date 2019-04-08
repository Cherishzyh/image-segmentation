from keras.preprocessing.image import ImageDataGenerator

# 数据预处理
# 读取图像文件，将jpge图像转换成RGB，将像素网格转换成浮点数张量，将像素缩放到0-1

# 从目录读取图像
train_data = ImageDataGenerator(rescale=1. / 255)
test_data = ImageDataGenerator(rescale=1. / 255)  # 将所有的图像乘1/255

train_generator = test_data.flow_from_directory(
    # 目标目录
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator = test_data.flow_from_directory(
    # 目标目录
    target_size=(150, 150),
    class_mode='binary',
    batch_size=20)
