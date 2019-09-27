import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from MeDIT.SaveAndLoad import LoadNiiData


def Showhist(image_array):
    plt.hist(image_array.flatten(), bins=50)
    plt.show()

# 画roi
def DrawNiiROI(data, roi, roi_save_path):
    for i in range(data.shape[-1]):
        one_image = os.path.join(roi_save_path, str(i) + '.png')
        plt.axis('off')
        plt.imshow(data[:, :, i], cmap='gray')
        plt.contour(roi[:, :, i], colors='g')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(one_image, bbox_inches='tight')
        plt.show()


