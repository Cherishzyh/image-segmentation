import matplotlib.pyplot as plt
import numpy as np


# evaluate loaded model on test data
def ModelEvaluate(model, image_test, label_test):
    score = model.evaluate(x=image_test, y=label_test)
    print('accuracy = ', score[1])
    return score


def Predict(model, test_image, prediction_store_folder):
    import os
    prediction = model.predict(test_image, verbose=0)
    print(prediction)
    np.save(os.path.join(prediction_store_folder, 'prediction.npy'), prediction)


def DrawROI(image_test, label_test):
    prediction = np.load('H:/data/Input_1_Output_1/prediction.npy')
    for i in range(len(prediction)):
        plt.contour(label_test[i, :, :, 0], colors='r')
        plt.contour(prediction[i, :, :, 0], colors='g')
        plt.imshow(image_test[i, :, :, 0], cmap='gray')
        plt.show()
        return 0


def Dice(y_true, y_pred):
    # parameter for loss function
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def TestDice(image_test, label_test):
    summary = 0.
    prediction = np.load('H:/data/Input_1_Output_1/prediction.npy')
    label = label_test.astype(np.float32)
    for i in range(len(image_test)):
        summary = summary + Dice(label[i], prediction[i])
    mean = summary / len(image_test)
    print("Mean Dice is : ", mean)
    return mean





