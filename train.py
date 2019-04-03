from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

# Load data
(x_image_train, y_label_train), (x_image_test, y_label_test) = cifar10.load_data()
x_image_train_normalize = x_image_train.astype('float32') / 255.0
x_image_test_normalize = x_image_test.astype('float32') / 255.0

# one-hot encoding.
from keras.utils import np_utils
y_label_train_one_hot = np_utils.to_categorical(y_label_train)
y_label_test_one_hot = np_utils.to_categorical(y_label_test)

from model import AlexNet
model = AlexNet((32, 32, 3))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

try:
    model.load_weights("C:/Users/I/.keras/SaveModel/cifarCnnModel.h5")
    print("加载模型成功！继续训练模型")
except:
    print("加载模型失败！开始训练一个新模型")


history = model.fit(x=x_image_train_normalize, y=y_label_train_one_hot, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

# save
model.save_weights("C:/Users/I/.keras/SaveModel/***********")
print("Save model to disk")
