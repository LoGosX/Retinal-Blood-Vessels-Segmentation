from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def mirror_sides(img, border_size):
    bs = border_size
    shape = np.array(img.shape)
    shape[:2] += 2 * bs
    r = np.zeros(shape, dtype=img.dtype)

    # interior
    r[bs:-bs, bs:-bs] = img

    # corners
    r[:bs, :bs] = img[:bs, :bs][::-1, ::-1]
    r[:bs, -bs:] = img[:bs, -bs:][::-1, ::-1]
    r[-bs:, :bs] = img[-bs:, :bs][::-1, ::-1]
    r[-bs:, -bs:] = img[-bs:, -bs:][::-1, ::-1]

    # sides
    r[:bs, bs:-bs] = img[:bs, :][::-1, :]
    r[-bs:, bs:-bs] = img[-bs:, :][::-1, :]
    r[bs:-bs, :bs] = img[:, :bs][:, ::-1]
    r[bs:-bs, -bs:] = img[:, -bs:][:, ::-1]

    return r


def train_val_test_split(*arrs, train_size=0.7, test_size=0.5, random_state):
    x_train, x_test, y_train, y_test, fov_train, fov_test = train_test_split(*arrs, train_size=train_size, random_state=random_state)
    x_val, x_test, y_val, y_test, fov_val, fov_test = train_test_split(x_test, y_test, fov_test, train_size=test_size, random_state=random_state)

    print(f"{len(x_train)=}")
    print(f"{len(x_val)=}")
    print(f"{len(x_test)=}")

    return x_train, x_val, x_test, y_train, y_val, y_test, fov_train, fov_val, fov_test


from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_augmented_data(x_train, y_train, seed, batch_size=1, augment=True):
    args = {
        'rotation_range': 10,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 5,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'reflect'
    }

    x_data_gen = ImageDataGenerator(**args)
    y_data_gen = ImageDataGenerator(**args)
    if augment:
        x_data_gen.fit(x_train, seed=seed)
        y_data_gen.fit(y_train, seed=seed)
    else:
        x_data_gen = ImageDataGenerator()
        y_data_gen = ImageDataGenerator()

    x_data_flow = x_data_gen.flow(x_train, shuffle=True, seed=seed, batch_size=batch_size)
    y_data_flow = y_data_gen.flow(y_train, shuffle=True, seed=seed, batch_size=batch_size)

    return zip(x_data_flow, y_data_flow)


def get_val_generator(x_val, y_val, seed, batch_size=1):
    x_val_flow = ImageDataGenerator().flow(x_val, shuffle=False, seed=seed, batch_size=batch_size)
    y_val_flow = ImageDataGenerator().flow(y_val, shuffle=False, seed=seed, batch_size=batch_size)

    val_gen = zip(x_val_flow, y_val_flow)

    return val_gen


def get_predictions(model, X, threshold=0.5):
    pred = np.vstack([model.predict(np.expand_dims(x, axis=0)) for x in X])
    if threshold is not None:
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0

    return pred


def patch_image(img, patch_size, stride):
    patches = []
    for r in range(0, img.shape[0], stride):
        if r - stride >= img.shape[0] - patch_size:
                continue
        for c in range(0, img.shape[1], stride):
            if c - stride >= img.shape[1] - patch_size:
                continue
            r = min(img.shape[0] - patch_size, r)
            c = min(img.shape[1] - patch_size, c)
            patches.append(img[r:r+patch_size, c:c+patch_size])
    return np.array(patches)

def from_patches(patches, org_shape, stride):
    reconstructed = np.zeros(org_shape, dtype="float32")
    n_overlaps = np.zeros_like(reconstructed)
    patch_size = patches[0].shape[0]
    patch_iter = iter(patches)
    for r in range(0, org_shape[0], stride):
        if r - stride >= org_shape[0] - patch_size:
            continue
        for c in range(0, org_shape[1], stride):
            if c - stride >= org_shape[1] - patch_size:
                continue
            r = min(org_shape[0] - patch_size, r)
            c = min(org_shape[1] - patch_size, c)
            reconstructed[r:r+patch_size, c:c+patch_size] += next(patch_iter)
            n_overlaps[r:r+patch_size, c:c+patch_size] += 1
    reconstructed /= n_overlaps
    return reconstructed

def plot_patches(patches, rows, cols, figsize=(20, 15), grayscale=False):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel()
    for ax, p in zip(axs, patches):
        ax.axis('off')
        if grayscale:
            ax.imshow(p, cmap='gray')
        else:
            ax.imshow(p)
    fig.show()


def plot_imgs(originals, ground_truths, predicted):
    n = len(originals)
    fig, axs = plt.subplots(n, 3, figsize=(20, 25))
    axs = axs.ravel()
    for i in range(n):
        axs[3 * i].imshow(originals[i])
        axs[3 * i + 1].imshow(ground_truths[i], cmap='gray')
        axs[3 * i + 2].imshow(predicted[i], cmap='gray')
        for ax, title in zip(axs[3 * i:3 * i + 3], ('original', 'ground truth', 'predicted')):
            ax.axis("off")
            ax.set_title(title)
    fig.tight_layout()


def get_predictions_from_patches(model, X, M, patch_size, stride, threshold=0.5):
    predictions = []
    for x, m in zip(X, M):
        patches_x = patch_image(x, patch_size, stride)
        patches_m = patch_image(m, patch_size, stride)
        pred_patches = []
        for patch_x, patch_m in zip(patches_x, patches_m):
            patch_x = np.expand_dims(patch_x, 0)
            patch_m = np.expand_dims(patch_m, 0)
            p = model.predict([patch_x, patch_m])[0]
            pred_patches.append(p)
        p = from_patches(np.asarray(pred_patches), x.shape[:2] + (1,), stride)
        predictions.append(p)
    predictions = np.asarray(predictions)



    if threshold is not None:
        predictions[predictions > threshold] = 1
        predictions[predictions <= threshold] = 0

    return predictions

# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
# https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
# https://arxiv.org/pdf/1706.05721.pdf <-- a=0.3 b=0.7
import tensorflow.keras.backend as K
def tversky_loss2(y_true, y_pred):
    alpha = 0.3
    beta  = 0.7
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

ALPHA = 0.3
BETA = 0.7

def tversky_loss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky