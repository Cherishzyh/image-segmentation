from keras.backend import flatten, sum
import keras.backend as K
import tensorflow as tf


#  metric function and loss function
def dice_coef(y_true, y_pred):
    # parameter for loss function
    smooth = 1
    y_true_f = flatten(y_true)
    y_pred_f = flatten(y_pred)
    intersection = sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def FocalLoss(y_true, y_pred, gamma=2., alpha=.25):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    eps = 1e-12
    y_pred = K.clip(y_pred, eps, 1. - eps)

    pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
    pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + 1e-6)) - \
           K.mean((1. - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-6))





