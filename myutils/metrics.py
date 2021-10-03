import sklearn.metrics as metrics
import numpy as np
from myutils.training import patch_image, from_patches, get_predictions_from_patches
from tqdm import trange
import tensorflow.keras.backend as K

def prediction_report(model, test_seq, patch_size, stride, threshold=0.5):
    report = {
        0: {
            'support': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        },
        1: {
            'support': 0,
            'support': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
    }
    
    for i in trange(len(test_seq)):
        (x, m), y = test_seq[i]
        org_shape = x.shape[1:]
        pred = get_predictions_from_patches(model, x, m, patch_size, stride, threshold)

        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        pred = np.uint8(pred)

        
        # does not work, oom
        # m = np.uint8(np.round(m))
        # mask_indices = np.argwhere(m == 1)
        # pred = pred[mask_indices]
        # y = y[mask_indices]

        

        if pred.ndim > y.ndim:
            y = np.expand_dims(y, -1)
        for class_num in (0, 1):
            opp_class = 1 - class_num
            tp = ((pred == class_num) & (y == class_num)).sum()
            tn = ((pred == opp_class) & (y == opp_class)).sum()
            fp = ((pred == class_num) & (y == opp_class)).sum()
            fn = ((pred == opp_class) & (y == class_num)).sum()
            report[class_num]['support'] += (y == class_num).sum()
            report[class_num]['tp'] += tp
            report[class_num]['tn'] += tn
            report[class_num]['fp'] += fp
            report[class_num]['fn'] += fn

        # pred has correct (masked) values outside fov, so it will definitely be true negative (tn)
        to_sub = m.size - m.sum()
        report[1]['tn'] -= to_sub
        report[0]['tp'] -= to_sub
        report[0]['support'] -= to_sub

    for class_num in (0, 1):
        tp, tn, fp, fn = report[class_num]['tp'], report[class_num]['tn'], report[class_num]['fp'], report[class_num]['fn']
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * recall * precision / (recall + precision)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        report[class_num] = {
            'support':report[class_num]['support'],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'g-mean': (sensitivity * specificity) ** .5,
            'tp':tp,
            'tn':tn,
            'fp':fp,
            'fn':fn
        }

    return report

def sensitivity_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity_metric(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def g_mean_metric(y_true, y_pred):
    return K.sqrt(sensitivity_metric(y_true, y_pred) * specificity_metric(y_true, y_pred))