#!/usr/bin/env python

import os
import glob
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from skimage.transform import resize
from skimage.io import imread, imsave
from models import *


def load_image_and_mask(image_path, mask_path, size=None):
    image = imread(image_path)[..., :3]
    if size is not None:
        img_h, img_w = image.shape[:2]
        scale = size / min(img_h, img_w)
        img_h, img_w = int(img_h * scale), int(img_w * scale)
        image = resize(image, (img_h, img_w))
    else:
        image = image.astype(np.float32) / 255.0
    if mask_path is not None:
        mask = imread(mask_path)
        mask = mask if mask.ndim == 3 else np.tile(mask[..., None], (1, 1, 3))
        if size is not None:
            mask = resize(mask, image.shape[:2])
    else:
        mask = np.ones_like(image)
    return image, mask


def decompose_albedo_normal_shadow(model, image, mask):
    albedo, normal, shadow = model.predict(image[None] * mask[None])
    albedo, normal, shadow = model.post_process(albedo, normal, shadow, mask)
    albedo, normal, shadow = albedo[0], normal[0], shadow[0]
    albedo = (albedo.numpy() * 255).astype(np.uint8)
    shadow = (shadow.numpy() * 255).astype(np.uint8)
    normal = (normal.numpy() * 128 + 128).astype(np.uint8)
    return albedo, normal, shadow


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images', '-i', type=pathlib.Path)
    parser.add_argument('--size', '-s', type=int, default=None)
    parser.add_argument('--ckpt',
                        '-c',
                        type=str,
                        default='pretrained/inverse_render_net/')
    args = parser.parse_args()

    inverse_render_net = InverseRenderNet()
    inverse_render_net.build(input_shape=(None, 200, 200, 3))
    inverse_render_net.load_weights(args.ckpt)

    for filename in glob.glob(os.path.join(args.images, '*.png')):
        print(f'processing {filename}')
        image, mask = load_image_and_mask(filename, None, size=args.size)
        albedo, normal, shadow = decompose_albedo_normal_shadow(
            inverse_render_net, image, mask)
        for s in ['albedo', 'normal', 'shadow']:
            os.makedirs(os.path.dirname(filename).replace('images', s),
                        exist_ok=True)
        imsave(filename.replace('images', 'albedo'), albedo)
        imsave(filename.replace('images', 'normal'), normal)
        imsave(filename.replace('images', 'shadow'), shadow)
