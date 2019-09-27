from CNNModel.CNNModel.Training.Generate import ImageInImageOut2D
from MeDIT.DataAugmentor import random_2d_augment
import numpy as np


generate = ImageInImageOut2D(r'C:\Users\yangs\Desktop\TZ roi\FormatH5', (240, 240), batch_size=1, augment_param=random_2d_augment)
for inputs, outputs in generate:
    print(inputs.shape)
    print(outputs.shape)

    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(np.squeeze(inputs), cmap='gray')
    plt.subplot(222)
    plt.imshow(np.squeeze(outputs[..., 0]), cmap='gray')
    plt.subplot(223)
    plt.imshow(np.squeeze(outputs[..., 1]), cmap='gray')
    plt.subplot(224)
    plt.imshow(np.squeeze(outputs[..., 2]), cmap='gray')
    plt.show()
