import os
import numpy as np

from CNNModel.Training.Generate import ImageInImageOut2DTest
from CNNModel.Utility.SaveAndLoad import LoadModel

from FilePath.Filepath import model_path, best_weights_path, store_folder


input_shape = (240, 240)


def SavePredict(image_test):
    model = LoadModel(model_path, best_weights_path)
    prediction = model.predict(image_test, verbose=1)
    np.save(os.path.join(store_folder, 'prediction_test.npy'), prediction[2])


def main():
    # load test data
    import matplotlib.pyplot as plt
    image_test, label_test, _ = ImageInImageOut2DTest(r'D:\ZYH\Data\TZ roi\Input_0_output_downsample_3\FormatH5\test', input_shape)
    SavePredict(image_test)

if __name__ == '__main__':
    main()


