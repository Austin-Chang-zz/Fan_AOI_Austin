import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Test TensorFlow by creating a simple computation graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
print("The result of a + b:", c.numpy())
