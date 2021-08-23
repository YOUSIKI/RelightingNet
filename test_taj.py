import os
from collections import namedtuple

import numpy as np
import tensorflow as tf
from skimage import io, transform
from tqdm import tqdm

from modules import irn_layer, lambSH_layer, renderingNet, sdNet, spade_models

# tensorflow flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', '0.001', "learning rate")
tf.app.flags.DEFINE_float('beta1', '0.5', "beta for Adam")
tf.app.flags.DEFINE_integer('batch_size', '5', "batch size")
tf.app.flags.DEFINE_integer('c_dim', '3', "c dimsion")
tf.app.flags.DEFINE_integer('z_dim', '64', "z dimsion")
tf.app.flags.DEFINE_integer('output_size', '200', "output size")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "Number of batches to run.")

# model paths
rendering_model_path = 'relight_model/model.ckpt'
skyGen_model_path = 'model_skyGen_net/model.ckpt'

taj_dir = "taj"


def new_size(img):
    max_const = 400
    h, w = img.shape[:2]
    scale = max_const / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return new_h, new_w


def resize_image(img, img_h, img_w):
    return transform.resize(img, (img_h, img_w))


# load images
images = dict()
for filename in tqdm(os.listdir(taj_dir), desc="Loading images"):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        if filename.find('_') == -1:
            name = '.'.join(filename.split('.')[:-1])
            image = io.imread(os.path.join(taj_dir, filename)) / 255.0
            img_h, img_w = new_size(image)
            image = resize_image(image, img_h, img_w)
            image = image[None, ...]
            images[name] = dict()
            images[name]['image'] = image
            images[name]['img_h'] = img_h
            images[name]['img_w'] = img_w
            images[name]['mask'] = np.ones_like(image)[..., :1]
            io.imsave(
                os.path.join(taj_dir, f'{name}_resized.png'),
                np.uint8(image[0] * 255.0),
            )

# decompose intrinsics
for name, image_dict in tqdm(images.items(), desc="Decomposing intrinsics"):
    img_h = image_dict['img_h']
    img_w = image_dict['img_w']

    # build model
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        input_var = tf.placeholder(tf.float32, (None, img_h, img_w, 3))
        mask_var = tf.placeholder(tf.float32, (None, img_h, img_w, 1))
        train_flag = tf.placeholder(tf.bool, ())
        input_noSky = input_var * mask_var
        input_shape = [5, img_h, img_w, 3]
        irnLayer = irn_layer.Irn_layer(input_shape, train_flag)
        albedo, shadow, nm_pred, lighting = irnLayer(input_noSky, mask_var)

    # restore vars
    irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='inverserendernet')
    rendering_vars = irn_vars
    sess = tf.InteractiveSession()
    rendering_saver = tf.train.Saver(rendering_vars)
    rendering_saver.restore(sess, rendering_model_path)

    # run
    albedo_var, shadow_var, nm_pred_var, lighting_var = sess.run(
        [albedo, shadow, nm_pred, lighting],
        feed_dict={
            input_var: image_dict['image'],
            mask_var: image_dict['mask'],
            train_flag: False
        })
    
    # close sess
    sess.close()

    # write to dict
    image_dict['albedo'] = albedo_var
    image_dict['shadow'] = shadow_var
    image_dict['nm_pred'] = nm_pred_var
    image_dict['lighting'] = lighting_var

    # save intrinsics
    io.imsave(os.path.join(taj_dir, f'{name}_albedo.png'),
              np.uint8(albedo_var[0] * 255.0))
    io.imsave(os.path.join(taj_dir, f'{name}_shadow.png'),
              np.uint8(shadow_var[0] * 255.0))
    io.imsave(os.path.join(taj_dir, f'{name}_nm_pred.png'),
              np.uint8((nm_pred_var[0] + 1) * 128.0))
    np.save(os.path.join(taj_dir, f'{name}_lighting.npy'), lighting_var[0])

# relighting (apply lighting2 to image1)
relighting_dict = dict()
with tqdm(total=len(images) * len(images), desc="Relighting") as pbar:
    for name1, image_dict1 in images.items():
        for name2, image_dict2 in images.items():
            # extract vars
            image_var = image_dict1['image']
            mask_var = image_dict1['mask']
            albedo_var = image_dict1['albedo']
            shadow_var = image_dict1['shadow']
            nm_pred_var = image_dict1['nm_pred']
            lighting_var = image_dict1['lighting']
            new_lighting_var = image_dict2['lighting']

            # build model
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=tf.AUTO_REUSE):
                train_flag = tf.placeholder(tf.bool, ())
                image = tf.placeholder(
                    tf.float32, (None, image_var.shape[1], image_var.shape[2],
                                 image_var.shape[3]))
                mask = tf.placeholder(tf.float32,
                                      (None, mask_var.shape[1],
                                       mask_var.shape[2], mask_var.shape[3]))
                albedo = tf.placeholder(
                    tf.float32, (None, albedo_var.shape[1],
                                 albedo_var.shape[2], albedo_var.shape[3]))
                shadow = tf.placeholder(
                    tf.float32, (None, shadow_var.shape[1],
                                 shadow_var.shape[2], shadow_var.shape[3]))
                nm_pred = tf.placeholder(
                    tf.float32, (None, nm_pred_var.shape[1],
                                 nm_pred_var.shape[2], nm_pred_var.shape[3]))
                lighting = tf.placeholder(
                    tf.float32,
                    (None, lighting_var.shape[1], lighting_var.shape[2]))
                new_lighting = tf.placeholder(tf.float32,
                                              (None, new_lighting_var.shape[1],
                                               new_lighting_var.shape[2]))
                shading, _ = lambSH_layer.lambSH_layer(tf.ones_like(albedo),
                                                       nm_pred, lighting,
                                                       tf.ones_like(shadow),
                                                       1.0)
                new_shading, _ = lambSH_layer.lambSH_layer(
                    tf.ones_like(albedo), nm_pred, new_lighting,
                    tf.ones_like(shadow), 1.0)
                rendering = tf.pow(albedo * shading * shadow, 1 / 2.2) * mask
                residual = (image * mask) - rendering
                OPTIONS = namedtuple('OPTIONS',
                                     ['output_c_dim', 'is_training', 'gf_dim'])
                options = OPTIONS(output_c_dim=1, is_training=True, gf_dim=8)

                def init_sd(nm_irn, lightings):
                    flatten_lightings = tf.tile(
                        tf.reshape(lightings, (-1, 1, 1, 27)),
                        (1, tf.shape(nm_irn)[1], tf.shape(nm_irn)[2], 1))

                    return tf.concat([nm_irn, flatten_lightings], axis=-1)

                shadow_gen = sdNet.shadow_generator(init_sd(
                    nm_pred, new_lighting),
                                                    options,
                                                    name='sd_generator')
                g_input = tf.concat(
                    [
                        albedo, nm_pred, new_shading, residual, shadow_gen,
                        1 - mask
                    ],
                    axis=-1,
                )
                relit_rendering = renderingNet.rendering_Net(
                    inputs=g_input,
                    masks=mask,
                    is_training=train_flag,
                    height=image_var.shape[1],
                    width=image_var.shape[2],
                    n_layers=30,
                    n_pools=4,
                    depth_base=32)
                init_sky = tf.random_normal(
                    (tf.shape(relit_rendering)[0], FLAGS.z_dim * 4),
                    dtype=tf.float32)
                cinput_sky1 = tf.image.resize_images(
                    (relit_rendering * 2. - 1.) * mask, (200, 200))
                cinput_sky2 = tf.image.resize_images(1. - mask, (200, 200))
                cinput_sky = tf.concat([cinput_sky1, cinput_sky2], axis=-1)
                sky = spade_models.generator(init_sky, cinput_sky, train_flag)
                sky = tf.image.resize_images(
                    sky, (image_var.shape[1], image_var.shape[2]))
                rendering = relit_rendering * mask + sky * (1. - mask)

            # restore vars
            irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope='inverserendernet')
            rn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='generator')
            sd_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='sd_generator')
            sg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='sky_generator')
            rendering_vars = rn_vars + irn_vars + sd_vars
            sess = tf.InteractiveSession()
            rendering_saver = tf.train.Saver(rendering_vars)
            rendering_saver.restore(sess, rendering_model_path)
            skyGen_saver = tf.train.Saver(sg_vars)
            skyGen_saver.restore(sess, skyGen_model_path)

            # run
            rendering_var = sess.run(rendering,
                                     feed_dict={
                                         train_flag: False,
                                         image: image_var,
                                         mask: mask_var,
                                         albedo: albedo_var,
                                         shadow: shadow_var,
                                         nm_pred: nm_pred_var,
                                         lighting: lighting_var,
                                         new_lighting: new_lighting_var,
                                     })
            
            # close sess
            sess.close()

            # write to dict
            name = f'{name1}_{name2}'
            relighting_dict[name] = rendering_var[0]

            # save image
            io.imsave(os.path.join(taj_dir, f'{name}.png'),
                      np.uint8(rendering_var[0] * 255.0))

            # update progress
            pbar.update(1)
