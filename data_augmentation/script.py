'''
1. 如何做扩增，Augmentation
2. 如何在训练过程中，做Augmentation
'''

#1.数据扩增
def AugmentScript():
    from MeDIT.SaveAndLoad import LoadNiiData
    from MeDIT.Visualization import Imshow3DArray, DrawBoundaryOfBinaryMask
    from MeDIT.Normalize import Normalize01
    import numpy as np
    import time

    t2_image, _, t2 = LoadNiiData(r'H:/data/TZ roi/BIAN ZHONG BEN/t2.nii', dtype=np.float32)
    roi_image, _, roi = LoadNiiData(r'H:/data/TZ roi/BIAN ZHONG BEN/prostate_roi_TrumpetNet.nii', dtype=np.uint8)

    Imshow3DArray(t2, ROI=roi)

    t2_slice = t2[..., 10]
    roi_slice = roi[..., 10]

    # DrawBoundaryOfBinaryMask(t2_slice, roi_slice)
    import matplotlib.pyplot as plt
    # plt.imshow(np.concatenate((Normalize01(t2_slice), Normalize01(roi_slice)), axis=1), cmap='gray')
    # plt.show()

    from MeDIT.DataAugmentor import DataAugmentor2D, AugmentParametersGenerator
    param_dict = {'stretch_x': 0.1, 'stretch_y': 0.1, 'shear': 0.1, 'rotate_z_angle': 20, 'horizontal_flip': True}

    augment_generator = AugmentParametersGenerator()
    augmentor = DataAugmentor2D()

    while True:
        augment_generator.RandomParameters(param_dict)
        transform_param = augment_generator.GetRandomParametersDict()
        print(transform_param)

        augment_t2 = augmentor.Execute(t2_slice, aug_parameter=transform_param, interpolation_method='linear')
        augment_roi = augmentor.Execute(roi_slice, aug_parameter=transform_param, interpolation_method='linear')
        # plt.imshow(np.concatenate((Normalize01(augment_t2), Normalize01(augment_roi)), axis=1), cmap='gray')
        # plt.show()

AugmentScript()

