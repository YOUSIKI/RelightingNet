import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from skimage.io import imread, imsave
from skimage.transform import resize

from models import *


def resize_img(img, shape):
    new_img_h, new_img_w = shape
    scale = img.shape[0] / new_img_h
    u, v = np.meshgrid(np.arange(new_img_w), np.arange(new_img_h))
    x = np.int32(u * scale)
    y = np.int32(v * scale)
    new_img = np.zeros((new_img_h, new_img_w, img.shape[-1]), np.float32)
    new_img[v, u] = img[y, x]
    return new_img


resize = resize_img

img1_path = "images/1.png"
img2_path = "images/2.png"
mask_path = "images/mask.png"

dst_dir = "test_timelapse"

img1 = imread(img1_path) / 255
img2 = imread(img2_path) / 255
mask = imread(mask_path) / 255
mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

img_h, img_w = img1.shape[:2]
scale = 200 / min(img_h, img_w)
img_h, img_w = int(img_h * scale), int(img_w * scale)

img1 = resize(img1, (img_h, img_w))
img2 = resize(img2, (img_h, img_w))
mask = resize(mask, (img_h, img_w))

img1 = tf.constant(img1, dtype=tf.float32, shape=(1, img_h, img_w, 3))
img2 = tf.constant(img2, dtype=tf.float32, shape=(1, img_h, img_w, 3))
mask = tf.constant(mask, dtype=tf.float32, shape=(1, img_h, img_w, 1))
img1_nosky = img1 * mask
img2_nosky = img2 * mask

irn = get_irn_layer(img_h, img_w, ckpt_path="pretrained/relight_model")
lambSH = get_lambSH_layer(img_h, img_w, gamma=1)
sdgen = get_shadow_generator(img_h, img_w)
rendering_net = get_rendering_net(img_h,
                                  img_w,
                                  ckpt_path="pretrained/relight_model/")
sky_generator = get_spade_generator(ckpt_path="pretrained/model_skyGen_net")

am1, nm1, sm1, ls1 = irn.predict([img1_nosky, mask])
am2, nm2, sm2, ls2 = irn.predict([img2_nosky, mask])

sd2, m2 = lambSH.predict([tf.ones_like(am2), nm2, tf.ones_like(sm2), ls2])
sd1, m1 = lambSH.predict([tf.ones_like(am1), nm1, tf.ones_like(sm1), ls1])

rendering = tf.pow(am2 * sd2 * sm2, 1 / 2.2) * mask
residual = img2_nosky - rendering

sd = sdgen.predict([nm2, ls1])

g_input = tf.concat([am2, nm2, sd1, residual, sd, 1 - mask], axis=-1)

relight_rendering = rendering_net.predict([g_input])

init_sky = tf.random.normal((relight_rendering.shape[0], 64 * 4),
                            dtype=tf.float32)
cinput_sky1 = tf.image.resize((relight_rendering * 2 - 1) * mask, (200, 200))
cinput_sky2 = tf.image.resize(1 - mask, (200, 200))
cinput_sky = tf.concat([cinput_sky1, cinput_sky2], axis=-1)
sky = sky_generator.predict([init_sky, cinput_sky])
sky = tf.image.resize(sky, (img_h, img_w))

relight_rendering = relight_rendering * mask + sky * (1 - mask)

os.makedirs(dst_dir, exist_ok=True)

rendering_val = np.uint8(relight_rendering[0] * 255)
albedo_val = np.uint8(am2[0] * 255)
shadow_val = np.uint8(sm2[0] * 255)
normal_val = np.uint8((nm2[0] + 1) * 128)
shading_val = np.uint8(sd1[0] * 255)

imsave(os.path.join(dst_dir, "rendering.png"), rendering_val)
imsave(os.path.join(dst_dir, "albedo.png"), albedo_val)
imsave(os.path.join(dst_dir, "shadow.png"), shadow_val)
imsave(os.path.join(dst_dir, "normal.png"), normal_val)
imsave(os.path.join(dst_dir, "shading.png"), shading_val)
