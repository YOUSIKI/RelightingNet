import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

__all__ = ["get_shadow_generator"]


def get_shadow_generator(image_h, image_w, ckpt_path=None):
    nm = keras.Input((image_h, image_w, 3))
    lightings = keras.Input((9, 3))
    shadow = shadow_generator(nm, lightings)
    model = keras.Model(inputs=[nm, lightings], outputs=shadow, name="sdgen")

    def ckpt_name_converter(name):
        if name.startswith("sd_generator"):
            return name.replace("sd_generator/", "").replace(
                "Conv/weights", "kernel") + ":0"
        else:
            return None

    if ckpt_path is not None:
        weights = dict()
        ckpt = tf.train.latest_checkpoint(ckpt_path)
        for name, shape in tf.train.list_variables(ckpt):
            new_name = ckpt_name_converter(name)
            if new_name is not None:
                weights[new_name] = tf.train.load_variable(ckpt, name)
        assert len(weights) == len(model.weights)
        for w in model.weights:
            assert w.name in weights, f"{w.name} not found"
            w.assign(weights[w.name])

    return model


def shadow_generator(nm, lightings, gf_dim=8):
    lightings = tf.reshape(lightings, (-1, 1, 1, 27))
    lightings = tf.tile(lightings, (1, nm.shape[1], nm.shape[2], 1))
    inputs = tf.concat([nm, lightings], axis=-1)
    e1 = tf.nn.relu(conv2d(inputs, gf_dim * 8, 3, 1, name="g_e1_conv"))
    e2 = max_pool2d(batch_norm(e1, name="g_bn_e1"), 2, 2, padding="SAME")
    e2 = tf.nn.relu(conv2d(e2, gf_dim * 16, 3, 1, name="g_e2_conv"))
    e3 = tf.nn.max_pool(batch_norm(e2, name="g_bn_e2"), 2, 2, padding="SAME")
    e3 = tf.nn.relu(conv2d(e3, gf_dim * 32, 3, 1, name="g_e3_conv"))
    e4 = tf.nn.max_pool(batch_norm(e3, name="g_bn_e3"), 2, 2, padding="SAME")
    e4 = tf.nn.relu(conv2d(e4, gf_dim * 64, 3, 1, name="g_e4_conv"))
    e5 = tf.nn.max_pool(batch_norm(e4, name="g_bn_e4"), 2, 2, padding="SAME")
    e5 = tf.nn.relu(conv2d(e5, gf_dim * 64, 3, 1, name="g_e5_conv"))
    e6 = tf.nn.max_pool(batch_norm(e5, name="g_bn_e5"), 2, 2, padding="SAME")
    e6 = tf.nn.relu(conv2d(e6, gf_dim * 64, 3, 1, name="g_e6_conv"))
    e7 = tf.nn.max_pool(batch_norm(e6, name="g_bn_e6"), 2, 2, padding="SAME")
    e7 = tf.nn.relu(conv2d(e7, gf_dim * 64, 3, 1, name="g_e7_conv"))
    e8 = tf.nn.max_pool(batch_norm(e7, name="g_bn_e7"), 2, 2, padding="SAME")
    e8 = tf.nn.relu(conv2d(e8, gf_dim * 64, 3, 1, name="g_e8_conv"))
    d1 = resize_bilinear(e8, e7.shape[1:3])
    d1 = conv2d(batch_norm(d1, name="g_sd_generator_bn_d1"),
                gf_dim * 64,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord1_dc")
    d1 = tf.concat([tf.nn.relu(d1), e7], 3)
    d2 = resize_bilinear(d1, e6.shape[1:3])
    d2 = conv2d(batch_norm(d2, name="g_sd_generator_bn_d2"),
                gf_dim * 64,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord2_dc")
    d2 = tf.concat([tf.nn.relu(d2), e6], 3)
    d3 = resize_bilinear(d2, e5.shape[1:3])
    d3 = conv2d(batch_norm(d3, name="g_sd_generator_bn_d3"),
                gf_dim * 64,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord3_dc")
    d3 = tf.concat([tf.nn.relu(d3), e5], 3)
    d4 = resize_bilinear(d3, e4.shape[1:3])
    d4 = conv2d(batch_norm(d4, name="g_sd_generator_bn_d4"),
                gf_dim * 64,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord4_dc")
    d4 = tf.concat([tf.nn.relu(d4), e4], 3)
    d5 = resize_bilinear(d4, e3.shape[1:3])
    d5 = conv2d(batch_norm(d5, name="g_sd_generator_bn_d5"),
                gf_dim * 32,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord5_dc")
    d5 = tf.concat([tf.nn.relu(d5), e3], 3)
    d6 = resize_bilinear(d5, e2.shape[1:3])
    d6 = conv2d(batch_norm(d6, name="g_sd_generator_bn_d6"),
                gf_dim * 16,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord6_dc")
    d6 = tf.concat([tf.nn.relu(d6), e2], 3)
    d7 = resize_bilinear(d6, e1.shape[1:3])
    d7 = conv2d(batch_norm(d7, name="g_sd_generator_bn_d7"),
                gf_dim * 8,
                kernel_size=3,
                strides=1,
                name="g_sd_generatord7_dc")
    d7 = tf.concat([tf.nn.relu(d7), e1], 3)
    d8 = resize_bilinear(d7, inputs.shape[1:3])
    d8 = conv2d(batch_norm(d8, name="g_sd_generator_bn_d8"),
                1,
                3,
                1,
                name="g_sd_generatord8_dc")
    pred = tf.clip_by_value(tf.nn.tanh(d8), -.9999, +.9999)
    pred = pred / 2 + 0.5
    return pred


def conv2d(x, filters, kernel_size, strides, padding="SAME", **kwargs):
    return layers.Conv2D(filters,
                         kernel_size,
                         strides,
                         padding,
                         use_bias=False,
                         **kwargs)(x)


def batch_norm(x, **kwargs):
    return layers.BatchNormalization(momentum=0.9,
                                     epsilon=1e-5,
                                     scale=True,
                                     **kwargs)(x)


def max_pool2d(x, kernel_size, strides, padding="SAME", **kwargs):
    return layers.MaxPool2D(kernel_size, strides, padding, **kwargs)(x)


def resize_bilinear(x, shape):
    return tf.image.resize(x, shape, method=tf.image.ResizeMethod.BILINEAR)


if __name__ == '__main__':
    from pprint import pprint
    model = get_shadow_generator(image_h=224,
                                 image_w=224,
                                 ckpt_path="pretrained/relight_model/")
    model.summary()
    pprint({w.name: w.shape for w in model.weights})