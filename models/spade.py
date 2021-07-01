import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

__all__ = ["get_spade_generator"]


def get_spade_generator(ckpt_path=None):
    noise = keras.Input(shape=(64 * 4))
    inputs = keras.Input(shape=(200, 200, 4))
    outputs = spade_generator(noise, inputs)
    model = keras.Model(inputs=[noise, inputs],
                        outputs=outputs,
                        name="sky_generator")

    def ckpt_name_converter(name):
        if name.startswith("sky_generator"):
            return name.replace("sky_generator/", "").replace(
                "/conv2d", "").replace("/dense", "") + ":0"
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


def spade_generator(noise, inputs):
    channels = 64 * 4 * 4
    x = layers.Dense(2 * 2 * channels, use_bias=True, name="linear_x")(noise)
    x = tf.reshape(x, (-1, 2, 2, channels))
    x = spade_resblock(inputs,
                       x,
                       channels,
                       use_bias=True,
                       name="spade_resblock_fix_0")
    shapes = [4, 7, 13, 25, 50, 100, 200]
    for i, shape in enumerate(shapes):
        x = upsample(x, shape, shape)
        if i > 2:
            channels = channels // 2
        x = spade_resblock(inputs,
                           x,
                           channels=channels,
                           use_bias=True,
                           name=f"spade_resblock_{i}")
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d(x, 3, 3, 1, pad=1, use_bias=True, name="logit")
    x = tf.nn.tanh(x)
    x = x / 2 + 0.5
    return x


def spade_resblock(segmap,
                   x_init,
                   channels,
                   use_bias=True,
                   spectral_norm=True,
                   name="spade_resblock"):
    channels_in = x_init.shape[-1]
    channels_mid = min(channels, channels_in)
    x = spade(segmap,
              x_init,
              channels_in,
              use_bias=use_bias,
              spectral_norm=False,
              name=name + "/spade_1")
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d(x,
               channels_mid,
               kernel_size=3,
               strides=1,
               pad=1,
               use_bias=use_bias,
               spectral_norm=spectral_norm,
               name=name + "/conv_1")
    x = spade(segmap,
              x,
              channels_mid,
              use_bias=use_bias,
              spectral_norm=False,
              name=name + "/spade_2")
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d(x,
               channels,
               kernel_size=3,
               strides=1,
               pad=1,
               use_bias=use_bias,
               spectral_norm=spectral_norm,
               name=name + "/conv_2")
    if channels_in != channels:
        x_init = spade(segmap,
                       x_init,
                       channels_in,
                       use_bias=use_bias,
                       spectral_norm=False,
                       name=name + "/spade_shortcut")
        x_init = conv2d(x_init,
                        channels,
                        kernel_size=1,
                        strides=1,
                        use_bias=False,
                        spectral_norm=spectral_norm,
                        name=name + "/conv_shortcut")
    return x + x_init


def spade(segmap,
          x_init,
          filters,
          use_bias=True,
          spectral_norm=True,
          name="spade"):
    b, h, w, c = x_init.shape
    x = param_free_norm(x_init)
    segmap_down = downsample(segmap, h, w)
    segmap_down = conv2d(segmap_down,
                         filters=128,
                         kernel_size=5,
                         strides=1,
                         pad=2,
                         use_bias=use_bias,
                         spectral_norm=spectral_norm,
                         name=name + "/conv_128")
    segmap_down = tf.nn.relu(segmap_down)
    segmap_gamma = conv2d(segmap_down,
                          filters=filters,
                          kernel_size=5,
                          strides=1,
                          pad=2,
                          use_bias=use_bias,
                          spectral_norm=spectral_norm,
                          name=name + "/conv_gamma")
    segmap_beta = conv2d(segmap_down,
                         filters=filters,
                         kernel_size=5,
                         strides=1,
                         pad=2,
                         use_bias=use_bias,
                         spectral_norm=spectral_norm,
                         name=name + "/conv_beta")
    x = x * (1 + segmap_gamma) + segmap_beta
    return x


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x_std = tf.sqrt(x_var + epsilon)
    return (x - x_mean) / x_std


def downsample(x, h, w):
    return tf.image.resize(x,
                           size=(h, w),
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def upsample(x, h, w):
    return tf.image.resize(x,
                           size=(h, w),
                           method=tf.image.ResizeMethod.BILINEAR)


def conv2d(x, *args, **kwargs):
    return Conv2D(*args, **kwargs)(x)


def spectral_norm(w, u):
    w_shape = w.shape
    w = tf.reshape(w, (-1, w.shape[-1]))
    u_hat = u
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = tf.nn.l2_normalize(v_)
    u_ = tf.matmul(v_hat, w)
    u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class Conv2D(layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 pad=0,
                 pad_mode="CONSTANT",
                 use_bias=True,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pad = pad
        self.pad_mode = pad_mode
        self.use_bias = use_bias
        self.spectral_norm = spectral_norm

    def build(self, input_shape):
        self.w = self.add_weight(name="kernel",
                                 shape=(self.kernel_size, self.kernel_size,
                                        input_shape[-1], self.filters))
        if self.use_bias:
            self.b = self.add_weight(name="bias", shape=(self.filters,))
        if self.spectral_norm:
            self.u = self.add_weight(name="u", shape=(1, self.filters))

    def call(self, x):
        if self.pad > 0:
            h = x.shape[1]
            if h % self.strides == 0:
                pad = self.pad * 2
            else:
                pad = max(0, self.kernel_size - h % self.strides)
            pad_t = pad // 2
            pad_b = pad - pad_t
            pad_l = pad // 2
            pad_r = pad - pad_l
            x = tf.pad(x, [[0, 0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]],
                       mode=self.pad_mode)
        w = self.w
        if self.spectral_norm:
            w = spectral_norm(w, self.u)
        x = tf.nn.conv2d(x, w, self.strides, padding="VALID")
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b)
        return x


if __name__ == "__main__":
    model = get_spade_generator(ckpt_path="pretrained/model_skyGen_net")
    # model.summary()
    # weight_shapes = {w.name: w.shape for w in model.weights}
    # from pprint import pprint
    # pprint(weight_shapes)
