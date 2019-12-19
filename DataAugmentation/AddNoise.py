import numpy as np
import h5py

from skimage.util import random_noise
from MeDIT.Normalize import Normalize01
import matplotlib.pyplot as plt


def GetH5(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        image = np.asarray(h5_file['input_0'], dtype=np.float32)
        label = np.asarray(h5_file['output_0'], dtype=np.uint8)
    return image, label


file_path = r'D:\ZYH\Data\TZ roi\TypeOfData\FormatH5\Problem\Chen ren geng-slicer_index_8.h5'
image, label = GetH5(file_path)

# normalize
new_data = Normalize01(image)
zero = np.zeros(shape=image.shape)

# add noise
image_noise = random_noise(new_data, mode='gaussian', var=0.0025)
# image_noise = random_noise(zero, mode='gaussian', var=0.09)

noise = image_noise - new_data

plt.subplot(121)
plt.title('original')
plt.axis('off')
plt.imshow(new_data, cmap='gray')

plt.subplot(122)
plt.title('add noise')
plt.axis('off')
plt.imshow(image_noise, cmap='gray')

plt.show()

var = np.var(noise)
mean = np.mean(noise)

print("The mean of noise is {}, var is {}".format(mean, var))
