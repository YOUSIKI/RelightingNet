import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

__all__ = ["get_lambSH_layer"]


def get_lambSH_layer(image_h, image_w, gamma=2.2):
    am = keras.Input((image_h, image_w, 3), batch_size=1)
    nm = keras.Input((image_h, image_w, 3), batch_size=1)
    sm = keras.Input((image_h, image_w, 1), batch_size=1)
    lightings = keras.Input((9, 3), batch_size=1)
    im, mask = lambSH_layer(am, nm, sm, lightings)
    model = keras.Model(inputs=[am, nm, sm, lightings],
                        outputs=[im, mask],
                        name="lambsh")
    return model


def lambSH_layer(am, nm, sm, lightings, gamma=2.2):
    mask = tf.not_equal(tf.reduce_sum(nm, axis=-1), 0)
    ones = tf.ones(nm.shape[0:3])
    ones = tf.expand_dims(ones, axis=-1)
    nm_homo = tf.concat([nm, ones], axis=-1)
    nm_homo = tf.expand_dims(nm_homo, axis=-1)
    nm_homo = tf.expand_dims(nm_homo, axis=-1)
    c1 = tf.constant(0.429043, dtype=tf.float32)
    c2 = tf.constant(0.511664, dtype=tf.float32)
    c3 = tf.constant(0.743125, dtype=tf.float32)
    c4 = tf.constant(0.886227, dtype=tf.float32)
    c5 = tf.constant(0.247708, dtype=tf.float32)
    M_row1 = tf.stack(
        [
            c1 * lightings[:, 8, :],
            c1 * lightings[:, 4, :],
            c1 * lightings[:, 7, :],
            c2 * lightings[:, 3, :],
        ],
        axis=1,
    )
    M_row2 = tf.stack(
        [
            c1 * lightings[:, 4, :],
            -c1 * lightings[:, 8, :],
            c1 * lightings[:, 5, :],
            c2 * lightings[:, 1, :],
        ],
        axis=1,
    )
    M_row3 = tf.stack(
        [
            c1 * lightings[:, 7, :],
            c1 * lightings[:, 5, :],
            c3 * lightings[:, 6, :],
            c2 * lightings[:, 2, :],
        ],
        axis=1,
    )
    M_row4 = tf.stack(
        [
            c2 * lightings[:, 3, :],
            c2 * lightings[:, 1, :],
            c2 * lightings[:, 2, :],
            c4 * lightings[:, 0, :] - c5 * lightings[:, 6, :],
        ],
        axis=1,
    )
    M = tf.stack([M_row1, M_row2, M_row3, M_row4], axis=1)
    M = tf.expand_dims(M, axis=1)
    M = tf.expand_dims(M, axis=1)
    E = tf.reduce_sum(nm_homo * M, axis=-3)
    E = tf.reduce_sum(E * nm_homo[..., 0, :], axis=-2)
    i = E * am * sm
    i = tf.clip_by_value(i, 0, 1) + 1e-4
    i = tf.pow(i, 1 / gamma)
    return i, mask


if __name__ == '__main__':
    from pprint import pprint
    model = get_lambSH_layer(image_h=224, image_w=224)
    model.summary()
    pprint({w.name: w.shape for w in model.weights})
