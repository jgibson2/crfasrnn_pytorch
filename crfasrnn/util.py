"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from PIL import Image

# Pascal VOC color palette for labels
_PALETTE = [0, 0, 0,
            128, 0, 0,
            0, 128, 0,
            128, 128, 0,
            0, 0, 128,
            128, 0, 128,
            0, 128, 128,
            128, 128, 128,
            64, 0, 0,
            192, 0, 0,
            64, 128, 0,
            192, 128, 0,
            64, 0, 128,
            192, 0, 128,
            64, 128, 128,
            192, 128, 128,
            0, 64, 0,
            128, 64, 0,
            0, 192, 0,
            128, 192, 0,
            0, 64, 128,
            128, 64, 128,
            0, 192, 128,
            128, 192, 128,
            64, 64, 0,
            192, 64, 0,
            64, 192, 0,
            192, 192, 0]



_IMAGENET_MEANS = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values


def get_preprocessed_image(file_name):
    """
    Reads an image from the disk, pre-processes it by subtracting mean etc. and
    returns a numpy array that's ready to be fed into the PyTorch model.

    Args:
        file_name:  File to read the image from

    Returns:
        A tuple containing:

        (preprocessed image, img_h, img_w, original width & height)
    """

    image = Image.open(file_name)
    original_size = image.size
    w, h = original_size
    ratio = min(500.0 / w, 500.0 / h)
    image = image.resize((int(w * ratio), int(h * ratio)), resample=Image.BILINEAR)
    im = np.array(image).astype(np.float32)
    assert im.ndim == 3, 'Only RGB images are supported.'
    im = im[:, :, :3]
    im = im - _IMAGENET_MEANS
    im = im[:, :, ::-1]  # Convert to BGR
    img_h, img_w, _ = im.shape

    pad_h = 500 - img_h
    pad_w = 500 - img_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return np.expand_dims(im.transpose([2, 0, 1]), 0), img_h, img_w, original_size


def get_label_image(probs, img_h, img_w, original_size, palette=_PALETTE):
    """
    Returns the label image (PNG with Pascal VOC colormap) given the probabilities.

    Args:
        probs:  Probability output of shape (num_labels, height, width)
        img_h:  Image height
        img_w:  Image width
        original_size: Original image size (width, height)

    Returns:
        Label image as a PIL Image
    """

    labels = probs.argmax(axis=0).astype('uint8')[:img_h, :img_w]
    label_im = Image.fromarray(labels, 'P')
    label_im.putpalette(palette)
    label_im = label_im.resize(original_size)
    return label_im

def label_image_to_values(label_image, palette=_PALETTE):
    PALETTE_MAP = {tuple(palette[i:i + 3]): i // 3 for i in range(0, len(palette), 3)}
    result = np.zeros(label_image.shape[0:2], dtype='uint8')
    for rgb, i in PALETTE_MAP.items():
        r,g,b = rgb
        result += np.uint8(i * np.where(label_image[:, :, 0] == r, 1, 0) * np.where(label_image[:, :, 1] == g, 1, 0) * np.where(label_image[:, :, 2] == b, 1, 0))
    return result


def compute_jaccard_index(label_img, ground_truth, palette=_PALETTE):
    img_a = label_image_to_values(label_img, palette)
    img_b = label_image_to_values(ground_truth, palette)
    return np.sum(np.logical_and(img_a == img_b, np.logical_or(img_a != 0, img_b != 0))) / np.sum(img_a + img_b > 0)