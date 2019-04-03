import matplotlib.pyplot as plt
from generator import GeneratorData

data_folder = 'H:/Input_1_Ouput_1/training'
batch_size = 12
GeneratorData(data_folder, batch_size)

for image_list, label_list in GeneratorData(data_folder, batch_size):
    for i in range(batch_size):
        plt.subplot(4, batch_size / 4, i + 1)
        plt.contour(label_list[i, :, :, 0], colors='r')
        plt.imshow(image_list[i, :, :, 0], cmap='gray')
    plt.show()

