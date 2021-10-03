import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

HRF_DATASET_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
DATASET_URL = HRF_DATASET_URL
DATASET_NAME = 'retinal_images'


def download_dataset(dataset_url: str = DATASET_URL, dataset_name: str = DATASET_NAME):
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname=dataset_name,
                                       extract=True)
    print(f"Dataset downloaded to: {pathlib.Path(data_dir)}")



def print_img_info(inp, tar, fov):
    print(f"""
Input:
shape {inp[0].shape} min(): {inp[0].min()}, max(): {inp[0].max()}, dtype: {inp[0].dtype}

Target:
shape {tar[0].shape} min(): {tar[0].min()}, max(): {tar[0].max()}, dtype: {tar[0].dtype}

Fov:
shape {fov[0].shape} min(): {fov[0].min()}, max(): {fov[0].max()}, dtype: {fov[0].dtype}
    """)


def load_images(inputs_path, targets_path, fovs_path, img_size=None):
    input_images_raw = [np.array(load_img(path, target_size=img_size), dtype="float32") for path in inputs_path]
    target_images_raw = [np.array(load_img(path, target_size=img_size, color_mode="grayscale"), dtype="float32") for
                         path in targets_path]
    fov_images_raw = [np.array(load_img(path, target_size=img_size, color_mode="grayscale"), dtype="float32") for path
                      in fovs_path]

    print_img_info(input_images_raw, target_images_raw, fov_images_raw)

    def f(img):
        return (img - img.min()) / (img.max() - img.min())

    input_images = [f(img) for img in input_images_raw]
    target_images = [np.expand_dims(f(img), axis=-1) for img in target_images_raw]
    fov_images = [np.expand_dims(f(img), axis=-1) for img in fov_images_raw]
    print_img_info(input_images, target_images, fov_images)

    input_images = np.asarray(input_images)
    target_images = np.asarray(target_images)
    fov_images = np.asarray(fov_images)

    return input_images, target_images, fov_images

def normalize_img(img):
    return img.astype("float32") / 255

def standarize_img(img):
    return (img - img.mean(axis=(0,1,2), keepdims=True)) / img.std(axis=(0,1,2), keepdims=True)

def process_images(x, y):
    x, y = np.asarray(x, dtype="float32"), np.asarray(y, dtype="float32")
    x = normalize_img(x)
    x = standarize_img(x)
    y = np.expand_dims(np.clip(y, 0, 1), -1)

    return x, y


def zca_whitening_matrix(X):
    """
    https://stackoverflow.com/a/38590790
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def prepare_for_pyplot(img):
    return (img - img.min()) / (img.max() - img.min())
