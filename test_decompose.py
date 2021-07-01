import os
import glob
import numpy as np
import tensorflow as tf
import shutil
from PIL import Image
from skimage import io
from skimage import transform
from collections import namedtuple
import argparse

from modules import renderingNet, sdNet, lambSH_layer, irn_layer, spade_models

rendering_model_path = 'relight_model/model.ckpt'
skyGen_model_path = 'model_skyGen_net/model.ckpt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', '0.001', """learning rate""")
tf.app.flags.DEFINE_float('beta1', '0.5', """beta for Adam""")
tf.app.flags.DEFINE_integer('batch_size', '5', """batch size""")
tf.app.flags.DEFINE_integer('c_dim', '3', """c dimsion""")
tf.app.flags.DEFINE_integer('z_dim', '64', """z dimsion""")
tf.app.flags.DEFINE_integer('output_size', '200', """output size""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")


def new_size(img):
    const = 400
    img_h, img_w = img.shape[:2]
    if img_h > img_w:
        scale = img_w / const
        new_img_h = np.int32(img_h / scale)
        new_img_w = const
    else:
        scale = img_h / const
        new_img_w = np.int32(img_w / scale)
        new_img_h = const

    return new_img_h, new_img_w


def resize_img(img, new_img_h, new_img_w):
    return transform.resize(img, (new_img_h, new_img_w))
    # scale = img.shape[0] / new_img_h
    # u, v = np.meshgrid(np.arange(new_img_w), np.arange(new_img_h))
    # x = np.int32(u * scale)
    # y = np.int32(v * scale)
    # new_img = np.zeros((new_img_h, new_img_w, 3), np.float32)
    # new_img[v, u] = img[y, x]
    # return new_img


def build_model(input_height, input_width):

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        input_var = tf.placeholder(tf.float32,
                                   (None, input_height, input_width, 3))
        mask_var = tf.placeholder(tf.float32,
                                  (None, input_height, input_width, 1))
        train_flag = tf.placeholder(tf.bool, ())
        input_noSky = input_var * mask_var
        input_shape = [5, input_height, input_width, 3]
        irnLayer = irn_layer.Irn_layer(input_shape, train_flag)
        albedo, shadow, nm_pred, lighting = irnLayer(input_noSky, mask_var)

        return albedo, shadow, nm_pred, input_var, mask_var, train_flag


images_path = './test_decompose'
output_path = './test_decompose_output_ver5'

os.makedirs(output_path, exist_ok=True)

for filename in glob.glob(os.path.join(images_path, '*.jpg')) + glob.glob(
        os.path.join(images_path, '*.png')):
    print(filename)
    img = io.imread(filename)
    new_img_h, new_img_w = new_size(img)
    img = resize_img(img / 255.0, new_img_h, new_img_w)
    mask = np.ones_like(img)
    mask = mask[..., :1]

    albedo_vars = np.zeros_like(img)
    normal_vars = np.zeros_like(img)
    counts_vars = np.zeros_like(img)

    albedo, shadow, nm_pred, input_var, mask_var, train_flag = build_model(
        200, 200)
    irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='inverserendernet')
    rendering_vars = irn_vars
    sess = tf.InteractiveSession()
    rendering_saver = tf.train.Saver(rendering_vars)
    rendering_saver.restore(sess, rendering_model_path)

    for i in range(0, new_img_h, 50):
        for j in range(0, new_img_w, 50):
            if i >= 0 and j >= 0 and i + 200 < new_img_h and j + 200 < new_img_w:
                crop_img = img[i:i + 200, j:j + 200, :]
                crop_mask = mask[i:i + 200, j:j + 200, :]

                albedo_var, shadow_var, nm_pred_var = sess.run(
                    [albedo, shadow, nm_pred],
                    feed_dict={
                        train_flag: False,
                        input_var: crop_img[None],
                        mask_var: crop_mask[None],
                    })

                albedo_vars[i:i + 200, j:j + 200, :] += albedo_var[0]
                normal_vars[i:i + 200, j:j + 200, :] += nm_pred_var[0]
                counts_vars[i:i + 200, j:j + 200, :] += 1

    for i in range(0, new_img_h, 50):
        for j in range(new_img_w, 0, -50):
            j = j - 200
            if i >= 0 and j >= 0 and i + 200 < new_img_h and j + 200 < new_img_w:
                crop_img = img[i:i + 200, j:j + 200, :]
                crop_mask = mask[i:i + 200, j:j + 200, :]

                albedo_var, shadow_var, nm_pred_var = sess.run(
                    [albedo, shadow, nm_pred],
                    feed_dict={
                        train_flag: False,
                        input_var: crop_img[None],
                        mask_var: crop_mask[None],
                    })

                albedo_vars[i:i + 200, j:j + 200, :] += albedo_var[0]
                normal_vars[i:i + 200, j:j + 200, :] += nm_pred_var[0]
                counts_vars[i:i + 200, j:j + 200, :] += 1

    for i in range(new_img_h, 0, -50):
        for j in range(0, new_img_w, 50):
            i = i - 200
            if i >= 0 and j >= 0 and i + 200 < new_img_h and j + 200 < new_img_w:
                crop_img = img[i:i + 200, j:j + 200, :]
                crop_mask = mask[i:i + 200, j:j + 200, :]

                albedo_var, shadow_var, nm_pred_var = sess.run(
                    [albedo, shadow, nm_pred],
                    feed_dict={
                        train_flag: False,
                        input_var: crop_img[None],
                        mask_var: crop_mask[None],
                    })

                albedo_vars[i:i + 200, j:j + 200, :] += albedo_var[0]
                normal_vars[i:i + 200, j:j + 200, :] += nm_pred_var[0]
                counts_vars[i:i + 200, j:j + 200, :] += 1

    for i in range(new_img_h, 0, -50):
        for j in range(new_img_w, 0, -50):
            i = i - 200
            j = j - 200
            if i >= 0 and j >= 0 and i + 200 < new_img_h and j + 200 < new_img_w:
                crop_img = img[i:i + 200, j:j + 200, :]
                crop_mask = mask[i:i + 200, j:j + 200, :]

                albedo_var, shadow_var, nm_pred_var = sess.run(
                    [albedo, shadow, nm_pred],
                    feed_dict={
                        train_flag: False,
                        input_var: crop_img[None],
                        mask_var: crop_mask[None],
                    })

                albedo_vars[i:i + 200, j:j + 200, :] += albedo_var[0]
                normal_vars[i:i + 200, j:j + 200, :] += nm_pred_var[0]
                counts_vars[i:i + 200, j:j + 200, :] += 1

    albedo_vars /= counts_vars
    normal_vars /= counts_vars

    albedo_var = np.uint8(albedo_vars * 255)
    normal_var = np.uint8((normal_vars + 1) * 128)
    basename = os.path.basename(filename)
    basename = basename[:-4] + "_%s" + ".png"
    io.imsave(os.path.join(output_path, basename % "image"), img)
    io.imsave(os.path.join(output_path, basename % "albedo"), albedo_var)
    io.imsave(os.path.join(output_path, basename % "normal"), normal_var)
