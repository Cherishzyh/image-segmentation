import os
import numpy as np

from CNNModel.Training.Generate import ImageInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
from CNNModel.Utility.SaveAndLoad import LoadModel

from Filepath import test_data_folder, model_path, best_weights_path,store_folder

input_shape = (240, 240, 1)
batch_size = 1


def SavePredict(image_test):
    model = LoadModel(model_path, best_weights_path)
    prediction = model.predict(image_test, batch_size)
    np.save(os.path.join(store_folder, 'prediction_test.npy'), prediction)


def main():
    # load test data
    image_test, label_test = ImageInImageOut2D(test_data_folder, input_shape,
                                               batch_size=batch_size, augment_param=random_2d_augment)
    SavePredict(image_test)
    prediction = np.load(os.path.join(store_folder, 'prediction_test.npy'))


if __name__ == '__main__':
    main()


