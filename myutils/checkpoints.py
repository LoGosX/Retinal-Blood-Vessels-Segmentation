from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from myutils.training import get_predictions_from_patches
from myutils.utils import process_images
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img


class PlotLearning(Callback):
    def __init__(self, metrics, figsize=(25, 15)):
        self._metrics = metrics
        self._figsize = figsize

    def on_train_begin(self, logs):
        self.i = 0
        self.x = []
        self.values = []
        self.fig = plt.figure()
        self.metrics = {m: [] for m in self._metrics}
        self.logs = []

    def on_epoch_end(self, epoch, logs):
        clear_output(wait=True)
        self.logs.append(logs)
        self.x.append(self.i)
        for k, v in self.metrics.items():
            v.append(logs.get(k))

        plt.figure(figsize=self._figsize)
        for metric in self.metrics:
            plt.plot(self.metrics[metric])

        # plt.yscale('log')
        #plt.ylim(0, 2.1)
        plt.title(f'Epoch #{epoch}')
        plt.ylabel('metrics')
        plt.xlabel('epoch')
        plt.legend(self.metrics.keys(), loc='upper right')

        plt.show()


class PredictAndSave(Callback):
    def __init__(self, x_file, patch_size, stride, every_n_batches=500, filename='predict_and_save.gif'):
        self._x_file = x_file
        self._filename = filename
        self._patch_size = patch_size
        self._stride = stride
        self._images = []
        self._every_n_batches = every_n_batches

    def on_batch_end(self, batch, logs):
        if batch % self._every_n_batches != 0:
            return
        x_to_predict = np.expand_dims(np.array(load_img(self._x_file)), 0)
        x_to_predict_processed = x_to_predict.astype("float32") / 255
        pred = get_predictions_from_patches(
            self.model, x_to_predict_processed, self._patch_size, self._stride, threshold=None)
        pred_to_save = np.squeeze(np.uint8(pred[0] * 255), -1)
        im = Image.fromarray(pred_to_save)
        self._images.append(im)

        def save(): return self._images[0].save(
            'animation.gif', save_all=True, append_images=self._images[1:], duration=250, loop=0)
        try:
            save()
        except KeyboardInterrupt:
            save()
            raise KeyboardInterrupt
