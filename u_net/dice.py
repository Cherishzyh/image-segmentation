from keras.backend import sum, flatten

# parameter for loss function
smooth = 1.


#  metric function and loss function
def dice_coef(y_true, y_pred):
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# load model
# model = load_model(weight_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
