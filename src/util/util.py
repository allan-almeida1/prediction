import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze_model(model: "any"):
    """
    Freezes the model

    Parameters
    ----------
    model : any
        The model to be frozen

    Returns
    -------
    any
        The frozen model
    """
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    return frozen_func


def predict_image(model: "any", image_path: "str", overlay=True, resolution=(640, 360)):
    """
    Predicts the path for a given image and shows the result

    Parameters
    ----------
    model : any
        The model to be used for prediction
    image_path : str
        The path to the image to be predicted
    overlay : bool, optional
        Whether to overlay the mask on the image, by default True
    resolution : tuple, optional
        The resolution to be used for the image, by default (640, 360)
    """
    img = plt.imread(image_path)
    image = cv2.resize(img, resolution) / 255
    image = np.array(image, dtype=np.float32)
    input_tensor = np.expand_dims(image, axis=0)
    mask_image = model.predict(input_tensor)[0]
    if not overlay:
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(image)
        ax[1].imshow(mask_image[:, :, 0], cmap="gray")
        plt.show()
    else:
        stack = np.zeros_like(mask_image, dtype=np.float32)
        mask = np.stack([0.5*mask_image, stack, mask_image], axis=2)
        mask = mask.reshape((resolution[1], resolution[0], 3))
        result = cv2.addWeighted(image, 1, mask, 1, 0)
        plt.imshow(result)


def feedforward(frozen_func: "any", image_path: "str", overlay=True, resolution=(640, 360)):
    """
    Predicts the path for a given image and shows the result

    Parameters
    ----------
    frozen_func : any
        The model to be used for prediction (frozen)
    image_path : str
        The path to the image to be predicted
    overlay : bool, optional
        Whether to overlay the mask on the image, by default True
    resolution : tuple, optional
        The resolution to be used for the image, by default (640, 360)
    """
    initial_time = time.time()
    img = plt.imread(image_path)
    image = cv2.resize(img, resolution) / 255
    image = np.array(image, dtype=np.float32)
    input_tensor = np.expand_dims(image, axis=0)
    input_tensor = tf.convert_to_tensor(input_tensor)
    mask_image = frozen_func(input_tensor)[0][0]
    print(f'Prediction time (ms): {(time.time() - initial_time)*1000}')
    if not overlay:
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(image)
        ax[1].imshow(mask_image[:, :, 0], cmap="gray")
        plt.show()
    else:
        stack = np.zeros_like(mask_image, dtype=np.float32)
        mask = np.stack([0.5*mask_image, stack, mask_image], axis=2)
        mask = mask.reshape((resolution[1], resolution[0], 3))
        result = cv2.addWeighted(image, 1, mask, 1, 0)
        plt.imshow(result)


def show_image_and_mask(image_path: "str"):
    """
    Shows an image and its mask

    Parameters
    ----------
    image_path : str
        The path to the image to be shown
    """
    img = plt.imread(image_path)
    mask_path = image_path.replace('.jpg', '_mask.jpg')
    mask = plt.imread(mask_path)
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img)
    ax[1].imshow(mask[:, :, 0], cmap="gray")
    plt.show()


def count_files(folders: "list"):
    """
    Counts the number of images and json files in a list of folders and prints the result

    Parameters
    ----------
    folders : list
        The list of folders to be counted

    Returns
    -------
    total_labeled : int
    """
    img_count = 0
    json_count = 0
    mask_count = 0
    for folder in folders:
        all_imgs = [name for name in os.listdir(
            folder) if name.endswith('.jpg') and not name.endswith('_mask.jpg')]
        all_masks = [name for name in os.listdir(
            folder) if name.endswith('_mask.jpg')]
        all_jsons = [name for name in os.listdir(
            folder) if name.endswith('.json')]
        imgs_labeled = [name for name in all_imgs if name.replace(
            '.jpg', '.json') in all_jsons or name.replace('.jpg', '_mask.jpg') in all_masks
            or 'jitter' in name or 'flip' in name]
        img_count += len(imgs_labeled)
        json_count += len(all_jsons)
        mask_count += len(all_masks)
        print((
            f'{folder}: {len(all_imgs)} path images, {len(all_jsons)} jsons, '
            f'{len(imgs_labeled)} labeled images, {len(all_masks)} mask images'
        ))
    print(
        f'Total: {img_count} labeled images, {json_count} jsons, {mask_count} mask images')
    return img_count


def get_labeled_files(folder: "str"):
    """
    Gets the list of labeled files in a folder

    Parameters
    ----------
    folder : str
        The folder to be searched

    Returns
    -------
    list
        The list of labeled files
    """
    all_imgs = [name for name in os.listdir(
        folder) if name.endswith('.jpg') and not name.endswith('_mask.jpg')]
    all_masks = [name for name in os.listdir(
        folder) if name.endswith('_mask.jpg')]
    all_jsons = [name for name in os.listdir(
        folder) if name.endswith('.json')]
    imgs_labeled = [name for name in all_imgs if name.replace(
        '.jpg', '.json') in all_jsons or name.replace('.jpg', '_mask.jpg') in all_masks
        or 'jitter' in name]
    return imgs_labeled


def get_label_and_masks(folder: "str"):
    """
    Gets the list of labeled files in a folder and their masks

    Parameters
    ----------
    folder : str
        The folder to be searched

    Returns
    -------
    (images, masks) : tuple
        The list of labeled files and the list of their masks
    """
    imgs_labeled = get_labeled_files(folder)
    imgs_labeled.sort()
    images = []
    masks = []
    for img_name in imgs_labeled:
        img_path = os.path.join(folder, img_name)
        if not 'jitter' in img_name:
            mask_path = img_path.replace('.jpg', '_mask.jpg')
        else:
            mask_path = img_path.replace('_jitter.jpg', '_mask.jpg')
        images.append(img_path)
        masks.append(mask_path)
    return (images, masks)


def json_2_mask(filepath: "str", show=True):
    """
    Reads a json file and returns the mask for the image and the image itself

    Parameters
    ----------
    filepath : str
        The path to the json file
    show : bool, optional
        Whether to show the image and the mask, by default True

    Returns
    -------
    (img, mask) : tuple
        The image and the mask
    """
    img = plt.imread(filepath.replace('.json', '.jpg'))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    with open(filepath) as f:
        data = json.load(f)
        shapes = data['shapes']
        for shape in shapes:
            points = shape['points']
            points = np.array(points, dtype=np.int32)
            mask = cv2.polylines(
                mask, [points], False, (255, 255, 255), int(0.035*img.shape[0]))
    if show:
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(img)
        ax[1].imshow(mask, cmap='gray')
        plt.show()
    return (img, mask)


def flip_mask(filepath: "str", show=True):
    """
    Reads a json file and returns the mask for the image and the image itself,
    flipping both horizontally

    Parameters
    ----------
    filepath : str
        The path to the json file
    show : bool, optional
        Whether to show the image and the mask, by default True

    Returns
    -------
    (img, mask) : tuple
        The image and the mask
    """
    img = plt.imread(filepath.replace('.json', '.jpg'))
    img = np.fliplr(img)
    _, mask = json_2_mask(filepath, show=False)
    mask = np.fliplr(mask)
    if show:
        _, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[1].imshow(mask, cmap='gray')
        ax[0].imshow(img)
        plt.show()
    return (img, mask)


def generate_mask_images(folders: "str"):
    """
    Generates the mask images for a list of folders

    Parameters
    ----------
    folders : str
        The list of folders to be processed
    """
    total_labeled = count_files(folders)
    total = 0
    for folder in folders:
        imgs_labeled = get_labeled_files(folder)
        print(f'{folder}: {len(imgs_labeled)} imagens rotuladas')
        print('Generating mask images...')
        for img_name in imgs_labeled:
            if 'jitter' in img_name or 'flip' in img_name or 'mask' in img_name:
                continue
            img_path = os.path.join(folder, img_name)
            json_path = os.path.join(folder, img_name.replace('.jpg', '.json'))
            _, mask = json_2_mask(json_path, False)
            mask_path = img_path.replace('.jpg', '_mask.jpg')
            cv2.imwrite(mask_path, mask)
            total += 1

    print('Found {} labeled images'.format(total_labeled))
    print('Generated {} mask images'.format(total))
    print('Done')
