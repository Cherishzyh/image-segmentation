import matplotlib.pyplot as plt


# draw one picture
def plot_image(image, size_inches_high, size_inches_width):
    fig = plt.gcf()
    fig.set_size_inches(size_inches_high, size_inches_width)
    plt.imshow(image, cmap='binary')
    plt.show()


# draw at most 25 pictures with label and prediction
def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        title = "label = " + str(labels[idx])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
