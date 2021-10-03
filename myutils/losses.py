from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K


class DiceLoss(Loss):
    def dice_coef(self, y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def call(self, y_true, y_pred):
        return self.dice_coef_loss(y_true, y_pred)


class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        alpha = self.alpha
        gamma = self.gamma
        pr = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss_1 = - y_true * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
        loss_0 = - (1 - y_true) * ((1 - alpha) * K.pow((pr), gamma) * K.log(1 - pr))
        loss = K.mean(loss_0 + loss_1)
        return loss
