import numpy as np
import os
from scipy import ndimage
import SimpleITK as sitk
import scipy.signal as signal

from CNNModel.SuccessfulModel.ConfigInterpretor import ConfigInterpretor, BaseImageOutModel
from MeDIT.ImageProcess import GetDataFromSimpleITK, GetImageFromArrayByImage
from MeDIT.Normalize import NormalizeForTensorflow
from MeDIT.SaveAndLoad import SaveNiiImage


class ProstatePzCgSegmentationInput_3(BaseImageOutModel):
    def __init__(self):
        super(ProstatePzCgSegmentationInput_3, self).__init__()
        self._image_preparer = ConfigInterpretor()

    def __KeepLargest(self, mask):
        new_mask = np.zeros(mask.shape)
        for position in range(1, 3):
            label_im, nb_labels = ndimage.label((mask == position).astype(int))
            max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
            index = np.argmax(max_volume)
            new_mask[label_im == index + 1] = position
        return new_mask

    def __FilterResult(self, mask):
        result = np.zeros_like(mask)
        for slice_index in range(mask.shape[-1]):
            one_slice = mask[..., slice_index]
            one_slice = signal.medfilt(one_slice, 5)
            result[..., slice_index] = one_slice
        return result


    def TransOneDataFor2_5DModel(self, data):
        # Here needs to be set according to config
        data_list = [data[..., :-2], data[..., 1:-1], data[..., 2:]]
        for input_data_index in range(len(data_list)):
            temp = data_list[input_data_index]
            temp = np.transpose(temp, (2, 0, 1))
            temp = temp[..., np.newaxis]
            temp = NormalizeForTensorflow(temp)
            data_list[input_data_index] = temp

        return data_list

    def invTransDataFor2_5DModel(self, preds):
        preds = np.squeeze(preds)
        preds = np.transpose(preds, (1, 2, 0))
        preds = np.concatenate((np.zeros((self._config.GetShape()[0], self._config.GetShape()[1], 1)),
                                preds,
                                np.zeros((self._config.GetShape()[0], self._config.GetShape()[1], 1))),
                               axis=-1)

        return preds


    def Run(self, image, store_folder=''):
        if isinstance(image, str):
            image = sitk.ReadImage(image)

        resolution = image.GetSpacing()
        flip_log = [0, 0, 0]
        _, data = GetDataFromSimpleITK(image, dtype=np.float32, is_flip=False, flip_log=flip_log)

        # Preprocess Data
        data = self._config.CropDataShape(data, resolution)
        input_list = self.TransOneDataFor2_5DModel(data)

        with self.graph.as_default():
            # 多监督模型输出三个，第一个为于是大小尺寸
            preds_list = self._model.predict(input_list)

        pred = preds_list[0]
        pred = np.argmax(pred, axis=-1)

        pred = self.invTransDataFor2_5DModel(pred)

        pred = self._config.RecoverDataShape(pred, resolution)
        pred = self.__FilterResult(pred)

        new_pred = self.__KeepLargest(np.round(pred))

        mask_image = GetImageFromArrayByImage(new_pred, image, flip_log=flip_log)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, '{}.nii.gz'.format(self._config.GetName()))
            SaveNiiImage(store_folder, mask_image)

        return new_pred, mask_image