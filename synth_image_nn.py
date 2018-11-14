import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import skimage.io
from scipy.misc import imresize
# from libs.utils import linear
from useful_stuff import linear
from libs import gif
from libs.utils import imcrop_tosquare

# performance stuff
import time
start_time = time.time()

# Save an image using the predicted y
def save_image(y_pred, name):
    clipped_img = np.clip(y_pred.reshape(in_image.shape).astype(np.float32), 0.0, 1.0)
    output_path = os.path.join(args.output_folder, name)
    plt.imsave(arr=clipped_img, fname=output_path)
    return clipped_img

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_image', type=str, help='the path to the input image', required=True)
parser.add_argument('-s', '--image_size', type=int, help='the size at which the input image will be resized')
parser.add_argument('-o','--output_folder', type=str, help='the path where to save the output image', required=True)
parser.add_argument('-b', '--batch_size', type=int, help='set the batch size')
parser.add_argument('--num_iterations', type=int, help='set the number of iterations')
parser.add_argument('-n', '--neurons', nargs='+', help='set the number of neurons of the network')
parser.add_argument('-g', '--make_gif', help='create a gif of the training process', action='store_true')
parser.add_argument('--save_iteration', help='save an image after every 2 iterations', action='store_true')
parser.add_argument('--ckpt_path', type=str, help='path where the model weights will be exported')
parser.add_argument('--load_ckpt', type=str, help='path from where to load a pretrained model')

args = parser.parse_args()

# Get the required arguments
INPUT_SIZE = 2
OUTPUT_SIZE = 3
image_size = args.image_size or 256
batch_size = args.batch_size or 200
num_iterations = args.num_iterations or 300
neurons = args.neurons or [INPUT_SIZE, image_size/2, image_size/2, image_size/2, image_size/2, image_size/2, image_size/2, OUTPUT_SIZE]
image_name = args.input_image.split('/')[-1].split('.')[-2]
if args.ckpt_path:
    export_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.ckpt_path)
    print('output model path: {}'.format(export_model_path))
if args.load_ckpt:
    load_model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.load_ckpt)

# Read the image
in_image = skimage.io.imread(args.input_image)
in_image = imcrop_tosquare(in_image)
in_image = imresize(in_image, (image_size, image_size))

print('successfully read image: {}'.format(args.input_image))
print('input image name: {}'.format(image_name))
print(in_image.shape)

x = []
y = []

# Get the positions and color from the image
for row in range(in_image.shape[0]):
    for col in range(in_image.shape[1]):
        # add positions
        x.append([row, col])
        # add color
        y.append(in_image[row][col])

x = np.array(x)
y = np.array(y)

# Normalize the inputs and outputs
x = (x - np.mean(x)) / np.std(x)
y = y / 255.0

# Init input and labels variables
X = tf.placeholder(shape=[None, x.shape[1]], dtype=tf.float32, name='X')
Y = tf.placeholder(shape=[None, y.shape[1]], dtype=tf.float32, name='Y')

current_input = X

# Create a fully connected network
for layer_i in range(1, len(neurons)):

    current_input = linear(
        X=current_input,
        n_input=neurons[layer_i-1],
        n_output=neurons[layer_i],
        activation=tf.nn.relu if (layer_i+1) < len(neurons) else None,
        scope='layer_{}'.format(layer_i)
    )
    Y_pred = current_input

    # h, W = linear(
    #     x=X,
    #     n_output=neurons[layer_i],
    #     name='linear_layer_{}'.format(layer_i),
    #     activation=tf.nn.relu if (layer_i+1) < len(neurons) else None,
    # )
        
    # Y_pred = tf.clip_by_value(h, 0, 255)

# MSE: ((y' - y)^2) / N
# cost = tf.reduce_mean(tf.squared_difference(Y_pred, Y))
# DISTANCE ERROR
error = tf.squared_difference(Y, Y_pred)
sum_error = tf.reduce_sum(error, 1) # compute the sum for each separate channel
cost = tf.reduce_mean(sum_error)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

num_batches = len(x) // batch_size

# if requested, store all the images in order to then make a gif
if args.make_gif:
    imgs = []

# if requested, prepare to save or load the model weights
if args.ckpt_path or args.load_ckpt:
    saver = tf.train.Saver(tf.global_variables())

# Train the model
with tf.Session() as sess:

    if not args.load_ckpt:

        # initialize global variables
        initialize_op = tf.global_variables_initializer()
        sess.run(initialize_op)

        # train for the required iterations
        for iter_i in range(num_iterations):
            print('{}/{}\r'.format(iter_i, num_iterations), end='')

            # randomize the current batch
            random_indices = np.random.permutation(range(len(x)))

            # Do mini batch stocastic gradient descent
            for batch_i in range(num_batches):
                start_index, end_index = batch_i * batch_size, (batch_i + 1) * batch_size
                current_batch_indices = random_indices[start_index : end_index]
                sess.run(optimizer, feed_dict={X:x[current_batch_indices], Y:y[current_batch_indices]})

            # compute training cost
            training_cost = sess.run(cost, feed_dict={X:x, Y:y})
            print('training cost: {}'.format(training_cost))

            # save an image for each iteration at the beginning (first 20 iterations)
            # and once in a while later
            if iter_i % 2 == 0:
                # compute predictions
                y_pred = sess.run(Y_pred, feed_dict={X:x, Y:y})
                
                if args.save_iteration:

                    image_output_name = '{}_batchsize{}_{}_{}.jpg'.format(image_name, batch_size, tf.nn.relu.__name__, iter_i)
                    clipped_img = save_image(y_pred, image_output_name)
                    print('{}, image saved'.format(iter_i))

                    if args.make_gif:
                        imgs.append(clipped_img)

        if args.ckpt_path:
            saved_path = saver.save(sess, os.path.join(export_model_path, args.ckpt_path.split(os.path.sep)[-1]))
            print('Model saved to: {}'.format(saved_path))

    else:
        saver.restore(sess, load_model_path)
        # compute predictions
        y_pred = sess.run(Y_pred, feed_dict={X:x, Y:y})

if args.make_gif:
    gif.build_gif(imgs, saveto=os.path.join(args.export_model_path, '{}.gif'.format(image_name)), show_gif=False)

save_image(y_pred, '{}_final.jpg'.format(image_name))
print('Finished!\nTime elapsed: --- {} seconds ---'.format(time.time() - start_time))