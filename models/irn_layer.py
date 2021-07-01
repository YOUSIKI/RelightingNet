import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

__all__ = ["get_irn_layer"]


def get_irn_layer(image_h, image_w, ckpt_path=None):
    inputs = keras.Input(shape=(image_h, image_w, 3))
    masks = keras.Input(shape=(image_h, image_w, 1))
    am_pred, nm_pred, sm_pred, lightings = irn_layer(inputs, masks)
    model = keras.Model(inputs=[inputs, masks],
                        outputs=[am_pred, nm_pred, sm_pred, lightings],
                        name="irn")

    def ckpt_name_converter(name):
        if name.startswith("inverserendernet"):
            return name.replace("inverserendernet/", "").replace(
                "Conv/weights", "kernel").replace("scale", "gamma").replace(
                    "offset", "beta") + ":0"
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


def irn_layer(inputs,
              masks,
              gf_dim=64,
              gamma=2.2,
              illupcapath="pretrained/illupca"):
    am_out, nm_out, sm_out = irn_resnet(inputs, gf_dim=gf_dim)
    am_pred = tf.nn.sigmoid(am_out) * masks + 1e-4
    sm_pred = tf.nn.sigmoid(sm_out) * masks + 1e-4
    nm_norm = tf.sqrt(tf.reduce_sum(nm_out**2, axis=-1, keepdims=True) + 1)
    nm_pred = tf.concat([nm_out / nm_norm, 1 / nm_norm], axis=-1) * masks
    illudecomp = IlluDecompLayer(illupcapath, gamma)
    lightings = illudecomp(inputs, masks, am_pred, nm_pred, sm_pred)
    return am_pred, nm_pred, sm_pred, lightings


class IlluDecompLayer(layers.Layer):

    def __init__(self, path='pretrained/illupca', gamma=2.2, **kwargs):
        super().__init__(**kwargs)
        self.vectors = load_numpy(os.path.join(path, 'vectors.npy'))
        self.means = load_numpy(os.path.join(path, 'means.npy'))
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def call(self, inputs, masks, am, nm, sm):
        shadings = tf.pow(inputs, self.gamma) / (am * sm)
        shadings = tf.clip_by_value(shadings, 0, 1) + 1e-4
        shadings = shadings * masks
        vars = tf.stack([shadings, nm], axis=-1)

        @tf.function
        def f(v):
            s = tf.reshape(v[..., 0], (-1, 3))
            n = tf.reshape(v[..., 1], (-1, 3))
            ones = tf.ones(tf.shape(n)[0:1])
            c1 = tf.constant(0.429043, dtype=tf.float32)
            c2 = tf.constant(0.511664, dtype=tf.float32)
            c3 = tf.constant(0.743125, dtype=tf.float32)
            c4 = tf.constant(0.886227, dtype=tf.float32)
            c5 = tf.constant(0.247708, dtype=tf.float32)
            A = tf.stack(
                [
                    c4 * ones,
                    2 * c2 * n[:, 1],
                    2 * c2 * n[:, 2],
                    2 * c2 * n[:, 0],
                    2 * c1 * n[:, 0] * n[:, 1],
                    2 * c1 * n[:, 1] * n[:, 2],
                    c3 * n[:, 2]**2 - c5,
                    2 * c1 * n[:, 2] * n[:, 0],
                    c1 * (n[:, 0]**2 - n[:, 1]**2),
                ],
                axis=-1,
            )
            return tf.matmul(pinv(A), s)

        lightings = tf.map_fn(f, vars)
        lightings = tf.reshape(lightings, (-1, 27))
        lightings_pca = tf.matmul((lightings - self.means), pinv(self.vectors))
        lightings = tf.matmul(lightings_pca, self.vectors) + self.means
        lightings = tf.reshape(lightings, (-1, 9, 3))
        return lightings


def irn_resnet(x, gf_dim=64):
    c0 = reflect_pad2d(x, 3)
    c1 = conv2d(c0, gf_dim, 7, 1, padding="VALID", name="g_e1_c")
    c1 = instance_norm(c1, name="g_e1_bn")
    c1 = tf.nn.relu(c1)
    c2 = conv2d(c1, gf_dim * 2, 3, 2, name="g_e2_c")
    c2 = instance_norm(c2, name="g_e2_bn")
    c2 = tf.nn.relu(c2)
    c3 = conv2d(c2, gf_dim * 4, 3, 2, name="g_e3_c")
    c3 = instance_norm(c3, name="g_e3_bn")
    c3 = tf.nn.relu(c3)
    r1 = residual_block(c3, gf_dim * 4, name="g_r1")
    r2 = residual_block(r1, gf_dim * 4, name="g_r2")
    r3 = residual_block(r2, gf_dim * 4, name="g_r3")
    r4 = residual_block(r3, gf_dim * 4, name="g_r4")
    r5 = residual_block(r4, gf_dim * 4, name="g_r5")
    r6 = residual_block(r5, gf_dim * 4, name="g_r6")
    r7 = residual_block(r6, gf_dim * 4, name="g_r7")
    r8 = residual_block(r7, gf_dim * 4, name="g_r8")
    r9 = residual_block(r8, gf_dim * 4, name="g_r9")
    am1 = resize_bilinear(r9, c2.shape[1:3])
    am1 = conv2d(am1, gf_dim * 2, 3, 1, name="g_am1_dc")
    am1 = tf.nn.relu(instance_norm(am1, name="g_am1_bn"))
    am2 = resize_bilinear(am1, c1.shape[1:3])
    am2 = conv2d(am2, gf_dim, 3, 1, name="g_am2_dc")
    am2 = tf.nn.relu(instance_norm(am2, name="g_am2_bn"))
    am2 = reflect_pad2d(am2, 3)
    am_out = conv2d(am2, 3, 7, 1, padding="VALID", name="g_am_out_c")
    nm1 = resize_bilinear(r9, c2.shape[1:3])
    nm1 = conv2d(nm1, gf_dim * 2, 3, 1, name="g_nm1_dc")
    nm1 = tf.nn.relu(instance_norm(nm1, name="g_nm1_bn"))
    nm2 = resize_bilinear(nm1, c1.shape[1:3])
    nm2 = conv2d(nm2, gf_dim, 3, 1, name="g_nm2_dc")
    nm2 = tf.nn.relu(instance_norm(nm2, name="g_nm2_bn"))
    nm2 = reflect_pad2d(nm2, 3)
    nm_out = conv2d(nm2, 2, 7, 1, padding="VALID", name="g_nm_out_c")
    sm1 = resize_bilinear(r9, c2.shape[1:3])
    sm1 = conv2d(sm1, gf_dim * 2, 3, 1, name="g_mask1_dc")
    sm1 = tf.nn.relu(instance_norm(sm1, name="g_mask1_bn"))
    sm2 = resize_bilinear(sm1, c1.shape[1:3])
    sm2 = conv2d(sm2, gf_dim, 3, 1, name="g_mask2_dc")
    sm2 = tf.nn.relu(instance_norm(sm2, name="g_mask2_bn"))
    sm2 = reflect_pad2d(sm2, 3)
    sm_out = conv2d(sm2, 1, 7, 1, padding="VALID", name="g_mask_out_c")
    return am_out, nm_out, sm_out


def residual_block(x, filters, kernel_size=3, strides=1, name=""):
    p = (kernel_size - 1) // 2
    y = reflect_pad2d(x, p)
    y = conv2d(y,
               filters,
               kernel_size,
               strides,
               padding="VALID",
               name=name + "_c1")
    y = instance_norm(y, name=name + "_bn1")
    y = reflect_pad2d(tf.nn.relu(y), p)
    y = conv2d(y,
               filters,
               kernel_size,
               strides,
               padding="VALID",
               name=name + "_c2")
    y = instance_norm(y, name=name + "_bn2")
    return y + x


def conv2d(x, filters, kernel_size, strides, padding="SAME", **kwargs):
    return layers.Conv2D(filters,
                         kernel_size,
                         strides,
                         padding,
                         use_bias=False,
                         **kwargs)(x)


def instance_norm(x, **kwargs):
    return tfa.layers.InstanceNormalization(epsilon=1e-5, **kwargs)(x)


def reflect_pad2d(x, p):
    return tf.pad(x, [
        [0, 0],
        [p, p],
        [p, p],
        [0, 0],
    ], 'reflect')


def resize_bilinear(x, shape):
    return tf.image.resize(x, shape, method=tf.image.ResizeMethod.BILINEAR)


def pinv(A, reltol=1e-6):
    s, u, v = tf.linalg.svd(A, compute_uv=True)
    atol = tf.reduce_max(s) * reltol
    s = tf.where(s > atol, s, atol * tf.ones_like(s))
    s_inv = tf.linalg.diag(1 / s)
    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


def load_numpy(filename):
    return tf.constant(np.load(filename), dtype=tf.float32)


if __name__ == '__main__':
    from pprint import pprint
    model = get_irn_layer(image_h=224,
                          image_w=224,
                          ckpt_path="pretrained/relight_model/")
    model.summary()
    pprint({w.name: w.shape for w in model.weights})