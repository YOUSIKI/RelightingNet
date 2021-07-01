import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

__all__ = ["get_rendering_net"]


def get_rendering_net(image_h, image_w, ckpt_path=None):
    inputs = keras.Input((image_h, image_w, 14))
    rendered = rendering_net(inputs)
    model = keras.Model(inputs=inputs, outputs=rendered, name="renderingnet")

    def ckpt_name_converter(name):
        if name.startswith("generator"):
            return name.replace(
                "generator/", "").replace("/biases", "/bias").replace(
                    "/group_norm/bias", "/group_norm/beta").replace(
                        "/group_norm/scale", "/group_norm/gamma").replace(
                            "/weights", "/kernel") + ":0"
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


def rendering_net(inputs, n_layers=30, n_pools=4, depth_base=32):
    conv_layers = n_layers // 2 - 1  # default 14
    deconv_layers = n_layers // 2  # default 15
    n_layers_before_pool = int(np.ceil(
        (conv_layers - 1) / n_pools) - 1)  # default 3
    n_pools_max = int(np.log2(512 / depth_base))  # default 4
    tail = conv_layers - n_layers_before_pool * n_pools_max  # default 2
    f_in_conv = [
        3, *[
            depth_base * int(2**np.ceil(i / n_layers_before_pool - 1))
            for i in range(1, conv_layers - tail + 1)
        ], *[
            depth_base * 2**n_pools_max
            for i in range(conv_layers - tail + 1, conv_layers + 1)
        ]
    ]
    f_out_conv = [
        64, *[
            depth_base * int(2**np.floor(i / n_layers_before_pool))
            for i in range(1, conv_layers - tail + 1)
        ], *[
            depth_base * 2**n_pools_max
            for i in range(conv_layers - tail + 1, conv_layers + 1)
        ]
    ]
    f_out_deconv = [*f_in_conv[:0:-1], 3]
    conv_out = inputs
    conv_out_list = list()
    for i in range(1, conv_layers + 2):
        name = f"conv{i}"
        filters = f_out_conv[i - 1]
        if (i - 1) % n_layers_before_pool == 0 and \
            i <= n_pools * n_layers_before_pool + 1 and \
                i != 1:
            conv_out_list.append(conv_out)
            conv_out = conv2d(conv_out, filters, name)
            conv_out = max_pool2d(conv_out, 2, 2, padding="SAME")
        else:
            conv_out = conv2d(conv_out, filters, name)
    deconv_out = conv_out
    for i in range(1, deconv_layers + 1):
        name = f"deconv{i}"
        filters = f_out_deconv[i - 1]
        if i % n_layers_before_pool == 0 and i <= n_pools * n_layers_before_pool:
            conv_out = conv_out_list[-(i // n_layers_before_pool)]
            deconv_out = conv2d(
                resize_bilinear(deconv_out, conv_out.shape[1:3]), filters, name)
            conv_out = conv2d(conv_out, filters, name + "/concat")
            deconv_out = deconv_out + conv_out
        elif i == deconv_layers:
            deconv_out = conv2d(deconv_out, filters, name, only_conv=True)
        else:
            deconv_out = conv2d(deconv_out, filters, name)
    return tf.clip_by_value(tf.nn.sigmoid(deconv_out), 1e-4, .9999)


def conv2d(x, filters, name, only_conv=False):
    x = layers.Conv2D(filters,
                      kernel_size=3,
                      strides=1,
                      padding="SAME",
                      use_bias=only_conv,
                      name=name)(x)
    if not only_conv:
        x = tfa.layers.GroupNormalization(32,
                                          epsilon=1e-5,
                                          name=name + "/group_norm")(x)
        x = tf.nn.relu(x)
    return x


def max_pool2d(x, ksize, strides, padding="SAME", **kwargs):
    return layers.MaxPool2D(ksize, strides, padding, **kwargs)(x)


def resize_bilinear(x, shape):
    return tf.image.resize(x, shape, method=tf.image.ResizeMethod.BILINEAR)


if __name__ == '__main__':
    model = get_rendering_net(224,
                              224,
                              ckpt_path="pretrained/relight_model/")
    model.summary()
    weight_shapes = {w.name: w.shape for w in model.weights}
    from pprint import pprint
    pprint(weight_shapes)