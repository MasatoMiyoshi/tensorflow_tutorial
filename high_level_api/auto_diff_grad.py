from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

def square(x):
    return tf.multiply(x, x)

grad = tfe.gradients_function(square)
print(square(3.))
print(grad(3.))

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
print(gradgrad(3.))

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
print(gradgradgrad(3.))

# With flow control:
def abs(x):
    return x if x > 0. else -x

grad = tfe.gradients_function(abs)
print(grad(3.))
print(grad(-3.))
### EOF
