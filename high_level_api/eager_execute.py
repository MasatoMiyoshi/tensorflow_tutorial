from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

"""
To start eager execution, add tf.enable_eager_execution()
to the beginning of the program or console session.
Do not add this operation to other modules that the program calls.
"""
print(tf.enable_eager_execution())

# Returns True if the current thread has eager execution enabled.
print(tf.executing_eagerly())

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2], [3, 4]])
print(a)

# Broadcasting support
b = tf.add(a, 1)
print(b)

# Operator overloading is supported
print(a * b)

c = np.multiply(a, b)
print(c)

# Obtain numpy value from a tensor:
print(a.numpy())
### EOF
