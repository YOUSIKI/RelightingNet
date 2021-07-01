import tensorflow as tf
import argparse
import pprint
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('weights converter: TensorFlow to Numpy')
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    args = parser.parse_args()
    return args


def main(args):
    vars = tf.train.list_variables(args.src)
    ckpt = tf.train.load_checkpoint(args.src)
    ckpt = {name: ckpt.get_tensor(name) for name, shape in vars}
    pprint.pprint(list(ckpt.keys()))
    np.savez_compressed(args.dst, **ckpt)


if __name__ == '__main__':
    main(parse_args())