import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers
import tensorflow_addons as tfa

__all__ = ['InverseRenderNet']


class InverseRenderNet(keras.Model):
    def __init__(self,
                 filters=64,
                 albedo_channels=3,
                 normal_channels=2,
                 shadow_channels=1):
        super().__init__()
        self.e1_conv = Conv2D(filters, 7, 1, 'valid', name='e1/conv')
        self.e1_norm = InstanceNorm2D(name='e1/norm')
        self.e2_conv = Conv2D(filters * 2, 3, 2, name='e2/conv')
        self.e2_norm = InstanceNorm2D(name='e2/norm')
        self.e3_conv = Conv2D(filters * 4, 3, 2, name='e3/conv')
        self.e3_norm = InstanceNorm2D(name='e3/norm')
        self.blocks = models.Sequential([
            ResidualBlock(filters * 4, name='r1'),
            ResidualBlock(filters * 4, name='r2'),
            ResidualBlock(filters * 4, name='r3'),
            ResidualBlock(filters * 4, name='r4'),
            ResidualBlock(filters * 4, name='r5'),
            ResidualBlock(filters * 4, name='r6'),
            ResidualBlock(filters * 4, name='r7'),
            ResidualBlock(filters * 4, name='r8'),
            ResidualBlock(filters * 4, name='r9'),
        ])
        self.a1_conv = Conv2D(filters * 2, 3, 1, name='a1/conv')
        self.a1_norm = InstanceNorm2D(name='a1/norm')
        self.a2_conv = Conv2D(filters, 3, 1, name='a2/conv')
        self.a2_norm = InstanceNorm2D(name='a2/norm')
        self.a3_conv = Conv2D(albedo_channels, 7, 1, 'valid', name='a3/conv')
        self.n1_conv = Conv2D(filters * 2, 3, 1, name='n1/conv')
        self.n1_norm = InstanceNorm2D(name='n1/norm')
        self.n2_conv = Conv2D(filters, 3, 1, name='n2/conv')
        self.n2_norm = InstanceNorm2D(name='n2/norm')
        self.n3_conv = Conv2D(normal_channels, 7, 1, 'valid', name='n3/conv')
        self.s1_conv = Conv2D(filters * 2, 3, 1, name='s1/conv')
        self.s1_norm = InstanceNorm2D(name='s1/norm')
        self.s2_conv = Conv2D(filters, 3, 1, name='s2/conv')
        self.s2_norm = InstanceNorm2D(name='s2/norm')
        self.s3_conv = Conv2D(shadow_channels, 7, 1, 'valid', name='s3/conv')

    def call(self, x):
        e0 = reflect_pad2d(x, 3)
        e1 = activations.relu(self.e1_norm(self.e1_conv(e0)))
        e2 = activations.relu(self.e2_norm(self.e2_conv(e1)))
        e3 = activations.relu(self.e3_norm(self.e3_conv(e2)))
        r9 = self.blocks(e3)
        a1 = resize_bilinear(r9, e2.shape[1:3])
        a1 = activations.relu(self.a1_norm(self.a1_conv(a1)))
        a2 = resize_bilinear(a1, e1.shape[1:3])
        a2 = activations.relu(self.a2_norm(self.a2_conv(a2)))
        a3 = reflect_pad2d(a2, 3)
        a3 = self.a3_conv(a3)
        n1 = resize_bilinear(r9, e2.shape[1:3])
        n1 = activations.relu(self.n1_norm(self.n1_conv(n1)))
        n2 = resize_bilinear(n1, e1.shape[1:3])
        n2 = activations.relu(self.n2_norm(self.n2_conv(n2)))
        n3 = reflect_pad2d(n2, 3)
        n3 = self.n3_conv(n3)
        s1 = resize_bilinear(r9, e2.shape[1:3])
        s1 = activations.relu(self.s1_norm(self.s1_conv(s1)))
        s2 = resize_bilinear(s1, e1.shape[1:3])
        s2 = activations.relu(self.s2_norm(self.s2_conv(s2)))
        s3 = reflect_pad2d(s2, 3)
        s3 = self.s3_conv(s3)
        return a3, n3, s3

    @staticmethod
    def post_process(albedo, normal, shadow, mask):
        def calculate_normal(pred):
            norm = tf.sqrt(tf.reduce_sum(pred**2, axis=-1, keepdims=True) + 1)
            pred = tf.concat([pred / norm, 1 / norm], axis=-1)
            return pred

        albedo = tf.sigmoid(albedo) * mask + 1e-4
        shadow = tf.sigmoid(shadow) * mask + 1e-4
        normal = calculate_normal(normal) * mask

        return albedo, normal, shadow


class ResidualBlock(keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters,
                            kernel_size,
                            strides,
                            padding='valid',
                            name='conv1')
        self.conv2 = Conv2D(filters,
                            kernel_size,
                            strides,
                            padding='valid',
                            name='conv2')
        self.norm1 = InstanceNorm2D(name='norm1')
        self.norm2 = InstanceNorm2D(name='norm2')
        self.pad = int((kernel_size - 1) / 2)

    def call(self, x):
        y = reflect_pad2d(x, self.pad)
        y = self.conv1(y)
        y = self.norm1(y)
        y = activations.relu(y)
        y = reflect_pad2d(y, self.pad)
        y = self.conv2(y)
        y = self.norm2(y)
        return y + x


class Conv2D(layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding='same',
                 **kwargs):
        super(Conv2D, self).__init__(
            filters,
            kernel_size,
            strides,
            padding,
            use_bias=False,
            kernel_initializer=initializers.GlorotUniform(),
            kernel_regularizer=regularizers.L2(1e-5),
            **kwargs,
        )


class InstanceNorm2D(tfa.layers.InstanceNormalization):
    def __init__(self, **kwargs):
        super(InstanceNorm2D, self).__init__(
            epsilon=1e-5,
            beta_initializer=initializers.RandomNormal(1.0, 0.02),
            gamma_initializer=initializers.Zeros(),
            **kwargs,
        )


def reflect_pad2d(x, pad):
    return tf.pad(x, [
        [0, 0],
        [pad, pad],
        [pad, pad],
        [0, 0],
    ], 'reflect')


def resize_bilinear(x, size):
    return tf.image.resize(
        x,
        size,
        method=tf.image.ResizeMethod.BILINEAR,
    )