from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

w = tfe.Variable([[1.0]])
with tfe.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, [w])
print(grad)

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
    return input * weight + bias

# A loss function using mean-squared error
def loss(weights, biases):
    error = prediction(training_inputs, weights, biases) - training_outputs
    return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
    with tfe.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value, [weights, biases])

def line_search_step(fn, init_x, rate=1.0):
    with tfe.GradientTape() as tape:
        # Variables are automatically recorded, but manually watch a tensor
        tape.watch(init_x)
        value = fn(init_x)
    grad, = tape.gradient(value, [init_x])
    grad_norm = tf.reduce_sum(grad * grad)
    init_value = value
    while value > init_value - rate * grad_norm:
        x = init_x - rate * grad
        value = fn(x)
        rate /= 2.0
    return x, value

train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
    dW, dB = grad(W, B)
    W.assign_sub(dW * learning_rate)
    B.assign_sub(dB * learning_rate)
    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
### EOF
