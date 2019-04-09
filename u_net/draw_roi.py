import matplotlib.pyplot as plt

from generator import GeneratorData

data_folder = 'H:/data/Input_1_Output_1/testing'
batch_size = 1
GeneratorData(data_folder, batch_size)

for image_list, label_list in GeneratorData(data_folder, batch_size):
    if batch_size == 1:
        for i in range(len(image_list)):
            plt.contour(label_list[i, :, :, 0], colors='r')
            plt.imshow(image_list[i, :, :, 0], cmap='gray')
        plt.show()
    else:
        for i in range(batch_size):
            plt.subplot(batch_size / 4, 4, i + 1)
            plt.contour(label_list[i, :, :, 0], colors='r')
            plt.imshow(image_list[i, :, :, 0], cmap='gray')
        plt.show()



