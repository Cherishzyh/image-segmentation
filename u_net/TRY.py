import numpy as np
from dice_loss import dice_coef
from filepath import test_data_folder
from data import GetData


image_test, label_test = GetData(test_data_folder)
dice = 0.
prediction = np.load('H:/data/SaveModel/model/prediction.npy')
label = label_test.astype(np.float32)
for i in range(len(image_test)):
    dice += dice_coef(label[i], prediction[i])
print('  ', dice)



# def TestDice():
#     dice = 0.
#     prediction = np.load('H:/data/Input_1_Output_1/prediction.npy')
#     label = label_test.astype(np.float32)
#     for i in range(len(image_test)):
#         dice += dice_coef(label[i], prediction[i])
#     dice = dice / len(image_test)
#     print("Mean Dice is : ", dice)
#     return dice
