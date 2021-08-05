import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import os

parser = argparse.ArgumentParser(
    usage="%(prog)s [OPTION] -I Input file",
    description="Sobel Edge Detection",
)
parser.add_argument(
    "-I", help="Input image", type=str
)
parser.add_argument(
    "-p", help="Percentile", type=float, default=99.7
)

args = parser.parse_args()

img = tf.io.read_file(args.I)
img = tf.image.decode_image(img, channels=3)
img = tf.image.rgb_to_grayscale(img)
img = tf.image.convert_image_dtype(img, tf.float32)
imgsob = tf.sqrt( tf.math.reduce_sum(tf.image.sobel_edges(tf.expand_dims(img, 0)) ** 2, axis = -1) )

pcentile = tfp.stats.percentile( tf.image.convert_image_dtype( imgsob[0], tf.uint8 ), args.p)

imgtest = tf.image.convert_image_dtype( imgsob[0], tf.uint8 )

imgtest = tf.cast( imgtest > pcentile, tf.uint8) * 255

tf.io.write_file(os.path.basename(args.I).split('.')[0] + "_ed.jpg", tf.image.encode_jpeg( imgtest ))
