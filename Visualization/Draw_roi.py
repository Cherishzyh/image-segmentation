import matplotlib.pyplot as plt
from DataProcess.Data import GeneratorData


def DrawROI(train_data_folder, batch_size):
    for image_list, label_list in GeneratorData(train_data_folder, batch_size):
        if batch_size == 1:
            for i in range(len(image_list)):
                plt.contour(label_list[i, :, :, 0], colors='r')
                plt.axis('off')
                plt.imshow(image_list[i, :, :, 0], cmap='gray')
            plt.show()
        else:
            for i in range(batch_size):
                plt.subplot(batch_size / 4, 4, i + 1)
                plt.axis('off')
                plt.contour(label_list[i, :, :, 0], colors='r')
                plt.imshow(image_list[i, :, :, 0], cmap='gray')
            plt.show()

train_data_folder = r'D:\ZYH\Data\TZ roi\TypeOfData\FormatH5\train'
batch_size = 4
for image_list, label_list in GeneratorData(train_data_folder, batch_size):
    if batch_size == 1:
        for i in range(len(image_list)):
            # plt.subplot(122)
            # plt.axis('off')
            # plt.imshow(label_list[i, :, :, 1], cmap='gray')
            # plt.subplot(121)
            plt.contour(label_list[i, :, :, 1], colors='r')
            plt.axis('off')
            plt.imshow(image_list[i, :, :], cmap='gray')
        plt.show()
    else:
        plt.figure(figsize=(18, 16))
        for i in range(batch_size):
            plt.subplot(batch_size / 2, 2, i + 1)
            plt.axis('off')
            plt.contour(label_list[i, :, :, 1], colors='r')
            plt.imshow(image_list[i, :, :], cmap='gray')
        plt.show()



