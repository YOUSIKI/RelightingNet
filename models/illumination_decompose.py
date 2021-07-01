import os
import numpy as np
import tensorflow as tf


class IlluminationDecompose(object):
    def __init__(self, path):
        self.gamma = tf.constant(2.2, dtype=tf.float32)

        def load_constant(name):
            return tf.constant(np.load(os.path.join(path, name + '.npy')),
                               dtype=tf.float32)

        self.vector = load_constant('vector')
        self.mean = load_constant('mean')
        self.var = load_constant('var')

    def __call__(self, image, albedo, normal, shadow, mask):
        shading = tf.pow(image, self.gamma) / (albedo * shadow)
        shading = (tf.clip_by_value(shading, 0, 1) + 1e-4) * mask
        outputs = list()
        c1 = tf.constant(0.429043, dtype=tf.float32)
        c2 = tf.constant(0.511664, dtype=tf.float32)
        c3 = tf.constant(0.743125, dtype=tf.float32)
        c4 = tf.constant(0.886227, dtype=tf.float32)
        c5 = tf.constant(0.247708, dtype=tf.float32)
        for i in range(image.shape[0]):
            shading_pixels = tf.reshape(shading[i], (-1, 3))
            normal_pixels = tf.reshape(normal[i], (-1, 3))
            ones = tf.ones(normal_pixels.shape[0], dtype=tf.float32)
            n0 = normal_pixels[:, 0]
            n1 = normal_pixels[:, 1]
            n2 = normal_pixels[:, 2]
            stack = [
                c4 * ones,
                2 * c2 * n1,
                2 * c2 * n2,
                2 * c2 * n0,
                2 * c1 * n0 * n1,
                2 * c1 * n1 * n2,
                c3 * n2**2 - c5,
                2 * c1 * n2 * n0,
                c1 * (n0**2 - n1**2),
            ]
            stack = tf.stack(stack, axis=-1)
            outputs.append(stack)
        return tf.stack(outputs, axis=0)
        # outputs.append(tf.matmul(self.pinv(stack), shading_pixels))
        # lighting = tf.reshape(tf.stack(outputs, axis=0), [-1, 27])
        # lighting_pca = tf.matmul((lighting - self.mean),
        #                          self.pinv(self.vector))
        # lighting = tf.matmul(lighting_pca, self.vector) + self.mean
        # lighting = tf.reshape(lighting, [-1, 9, 3])
        # return lighting

    @staticmethod
    def pinv(mat, reltol=1e-6):
        s, u, v = tf.linalg.svd(mat)
        atol = tf.reduce_max(s) * reltol
        s = tf.where(s > atol, s, atol * tf.ones_like(s))
        s_inv = tf.linalg.diag(1 / s)
        return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))
