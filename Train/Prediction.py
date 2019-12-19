import os
import numpy as np
import matplotlib.pyplot as plt

from CNNModel.Training.Generate import ImageInImageOut2DTest
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01
from CNNModel.SuccessfulModel.ProstateSegment import ProstatePzCgSegmentation, ProstatePzCgSegmentationInput_3
from MeDIT.SaveAndLoad import LoadNiiData

from FilePath.Filepath import model_path, test_folder, validation_folder, train_folder, prediction_folder


# evaluate loaded model on test data
def ModelEvaluate(model, image_test, label_test):
    score = model.evaluate(x=image_test, y=label_test)
    print('accuracy = ', score[1])
    return score


def Predict(model, test_image, prediction_store_folder):
    import os
    prediction = model.predict(test_image, verbose=0)
    np.save(os.path.join(prediction_store_folder, 'prediction_train.npy'), prediction)


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


def ArgMax(one_label):
    argmax_one_label = np.zeros(shape=one_label.shape)
    for raw in range(one_label.shape[0]):
        for column in range(one_label.shape[1]):
            index = np.argmax(one_label[raw, column])
            argmax_one_label[raw, column, index] = 1
    return argmax_one_label


def ShowHist(data, title=None, save_path=None):
    plt.hist(data)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def ShowRoi(image_test, label_test, prediction):
    for index in range(prediction.shape[0]):
        pred_binary = ArgMax(prediction[index])

        plt.figure(figsize=(16, 8))

        plt.subplot(121)
        plt.axis('off')
        plt.title('Label')
        plt.contour(label_test[index, :, :, 1], colors='r')
        plt.imshow(image_test[index, :, :, 0], cmap='gray')

        plt.subplot(122)
        plt.axis('off')
        plt.title('pred')
        plt.contour(pred_binary[:, :, 1], colors='b')
        plt.imshow(image_test[index, :, :, 0], cmap='gray')

        plt.show()


def ComputeDice(image_test, label_test, prediction):
    dice_pz_list = []

    dice_tz_list = []

    for index in range(prediction.shape[0]):
        pred_binary = ArgMax(prediction[index])
        dice_pz_list.append(Dice(label_test[index, :, :, 1], pred_binary[:, :, 1]))
        dice_tz_list.append(Dice(label_test[index, :, :, 2], pred_binary[:, :, 2]))
    #     dice_list.append(Dice(label_test[index, :, :, channel], pred_binary[:, :, channel]))
    #
    # print('dice', sum(dice_list) / (prediction.shape[0] * 3))
    # print()
    # ShowHist(dice_list, title='dice of test',
    #          save_path=os.path.join(r'/home/zhangyihong/Documents/TZ_ROI/Model_Add65/Dice', 'dice of test.jpg'))

    print('pz dice', sum(dice_pz_list) / (prediction.shape[0]))
    print()

    ShowHist(dice_pz_list, title='pz dice of train',
             save_path=os.path.join(r'/home/zhangyihong/Documents/TZ_ROI/DSNModel_initail/Model_Focal_Loss/dice', 'pz dice of train.jpg'))

    print('tz dice', sum(dice_tz_list) / (prediction.shape[0]))
    print()
    ShowHist(dice_tz_list, title='tz dice of train',
             save_path=os.path.join(r'/home/zhangyihong/Documents/TZ_ROI/DSNModel_initail/Model_Focal_Loss/dice', 'tz dice of train.jpg'))

    show_roi = True
    if show_roi:
        ShowRoi(image_test, label_test, prediction)


def Compute3DDice(prediction, image_test, label_test, case_name):
    image_list = []
    pred_list = []
    label_list = []
    pz_dice = []
    tz_dice = []

    for case_num in range(prediction.shape[0]):
        print(case_num)
        index = case_name[case_num].index('-')
        if case_num == 0 or case_name[case_num][0: index] == case_name[case_num-1][0: index] and case_num != prediction.shape[0]-1:
            pred_label = ArgMax(prediction[case_num])
            pred_list.append(pred_label)
            label_list.append(label_test[case_num])
            image_list.append(image_test[case_num])

        elif case_num == prediction.shape[0]-1:
            print('last')
            pred = np.asarray(pred_list)
            label = np.asarray(label_list)
            pz_dice.append(Dice(label[..., 1], pred[..., 1]))
            tz_dice.append(Dice(label[..., 2], pred[..., 2]))
            pred_list = []
            label_list = []
        else:
            print('next')
            pred = np.asarray(pred_list)
            label = np.asarray(label_list)
            image = np.asarray(image_list)

            pz_dice.append(Dice(label[..., 1], pred[..., 1]))
            tz_dice.append(Dice(label[..., 2], pred[..., 2]))
            pred_list = []
            label_list = []
            image_list = []

            pred_label = ArgMax(prediction[case_num])
            pred_list.append(pred_label)
            label_list.append(label_test[case_num])
            image_list.append(image_test[case_num])

    print('pz_dice:', pz_dice)
    print('tz_dice:', tz_dice)


def PredictNii():
    model_folder_path = model_path
    t2_folder = test_folder
    store_folder = r'D:\test'
    t2_list = os.listdir(t2_folder)

    prostate_segmentor = ProstatePzCgSegmentationInput_3()
    prostate_segmentor.LoadConfigAndModel(model_folder_path)

    for case in t2_list:
        print('Predicting ', case)
        case_path = os.path.join(t2_folder, case)
        store_path = os.path.join(store_folder, case)
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        t2_path = os.path.join(case_path, 't2.nii')

        t2_image, _, t2_data = LoadNiiData(t2_path, dtype=np.float32, is_show_info=False)

        prostate_segmentor.Run(t2_image, store_folder=os.path.join(store_path, 'prediction.nii'))


def main():
    # PredictNii()
    t2_folder = test_folder
    predict_folder = r'Y:\142\zhangyihong\Documents\TZ_ROI\DSN_Add65_3Input\input3_output0_down\prediction'
    t2_list = os.listdir(t2_folder)

    cg_dice_list = []
    pz_dice_list = []

    for case in t2_list:
        case_path = os.path.join(t2_folder, case)
        predict_path = os.path.join(predict_folder, case)

        t2_path = os.path.join(case_path, 't2.nii')
        merge_roi_path = os.path.join(case_path, 'merge_pz1_cg2.nii')
        predict_roi_path = os.path.join(predict_path, 'prediction.nii')

        t2_image, _, t2_data = LoadNiiData(t2_path, dtype=np.float32, is_show_info=False)
        _, _, roi_data = LoadNiiData(merge_roi_path, dtype=np.uint8, is_show_info=False)
        _, _, predict_roi_data = LoadNiiData(predict_roi_path, dtype=np.uint8, is_show_info=False)

        cg_dice_list.append(Dice((roi_data == 2).astype(int), (predict_roi_data == 2).astype(int)))
        pz_dice_list.append(Dice((roi_data == 1).astype(int), (predict_roi_data == 1).astype(int)))
        print('cg dice:', Dice((roi_data == 2).astype(int), (predict_roi_data == 2).astype(int)))
        print('pz dice:', Dice((roi_data == 1).astype(int), (predict_roi_data == 1).astype(int)))

        # from MeDIT.Visualization import Imshow3DArray
        # from MeDIT.Normalize import Normalize01
        #
        # Imshow3DArray(Normalize01(t2_data), roi=(roi_data == 1).astype(int))
        # Imshow3DArray(Normalize01(t2_data), roi=[(predict_roi_data == 1).astype(int), (roi_data == 1).astype(int)])
    ShowHist(cg_dice_list, save_path=os.path.join(model_path, 'cg_dice_hist.jpg'), title='dice of cg')
    ShowHist(pz_dice_list, save_path=os.path.join(model_path, 'pz_dice_hist.jpg'), title='dice of pz')

    print('mean dice of cg:', sum(cg_dice_list) / len(cg_dice_list))
    print('mean dice of pz:', sum(pz_dice_list) / len(pz_dice_list))


if __name__ == '__main__':
    main()


