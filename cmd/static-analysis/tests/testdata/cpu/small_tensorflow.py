import tensorflow as tf



if __name__ == "__main__":
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[3.0], [4.0]])
        c = tf.matmul(a, b)  # This runs on GPU 0