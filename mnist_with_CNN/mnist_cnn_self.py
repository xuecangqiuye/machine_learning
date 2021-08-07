import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
test_acc = []


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def inference(x, keep_prob):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope('conv_layer1'):
        weights = weight_variable([5, 5, 1, 32])
        biases = bias_variable([32])
        layer1 = tf.nn.relu(conv2d(x_image, weights) + biases)
        layer1_pool = max_pool_2x2(layer1)

    with tf.variable_scope('conv_layer2'):
        weights = weight_variable([5, 5, 32, 64])
        biases = bias_variable([64])
        layer2 = tf.nn.relu(conv2d(layer1_pool, weights) + biases)
        layer2_pool = max_pool_2x2(layer2)

    with tf.variable_scope('fc_layer3'):
        weights = weight_variable([7 * 7 * 64, 1024])
        biases = bias_variable([1024])
        layer2_reshape = tf.reshape(layer2_pool, [-1, 7 * 7 * 64])
        layer3 = tf.nn.relu(tf.matmul(layer2_reshape, weights) + biases)

    with tf.variable_scope('drop_out'):
        layer3_drop = tf.nn.dropout(layer3, keep_prob)

    with tf.variable_scope('softmax_layer4'):
        weights = weight_variable([1024, 10])
        biases = bias_variable([10])

        y_softmax = tf.nn.softmax(tf.matmul(layer3_drop, weights) + biases)

    return y_softmax


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    keep_prob = tf.placeholder("float")
    y_softmax = inference(x, keep_prob)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_softmax))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_softmax, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(5000):
        xs, ys = mnist.train.next_batch(32)
        train_step.run(session=sess, feed_dict={x: xs, y_: ys, keep_prob: 0.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess, feed_dict={x: xs, y_: ys, keep_prob: 1.0})
            print("step {0}, train_accuracy {1}".format(i, train_accuracy))
    for i in range(mnist.test.num_examples):
        start = (i * 1000) % mnist.test.num_examples
        end = min(start + 1000, mnist.test.num_examples)
        xt = mnist.test.images[start:end]
        yt = mnist.test.labels[start:end]
        print("test group [{0}:{1}] test accuracy {2}".format(start, end, accuracy.eval(session=sess,
                                                                                      feed_dict={x: xt, y_: yt,
                                                                                                 keep_prob: 1.0})))
        test_acc.append(accuracy.eval(session=sess, feed_dict={x: xt, y_: yt, keep_prob: 1.0}))
    print('test accuracy is {0}'.format(sess.run(tf.reduce_mean(test_acc))))


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    start = time.clock()
    train(mnist)
    end = time.clock()  # 计算程序结束时间
    print("running time is {0} s".format(end - start))


if __name__ == "__main__":
    tf.app.run()
