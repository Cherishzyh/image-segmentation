from data import GetData
from read_model import ReadModel
from filepath import test_data_folder, model_path, last_weights_path, prediction_store_folder
import numpy as np

# load test data
image_test, label_test = GetData(test_data_folder)

# load model
model = ReadModel(model_path, last_weights_path)


def Predict():
    import os
    prediction = model.predict(image_test, verbose=0)
    print(prediction)
    np.save(os.path.join(prediction_store_folder, 'prediction.npy'), prediction)


def Dice(y_true, y_pred):
    # parameter for loss function
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def TestDice():
    summary = 0.
    prediction = np.load('H:/data/Input_1_Output_1/prediction.npy')
    label = label_test.astype(np.float32)
    for i in range(len(image_test)):
        summary = summary + Dice(label[i], prediction[i])
    mean = summary / len(image_test)
    print("Mean Dice is : ", mean)
    return mean


# Predict()
# TestDice()
