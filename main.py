import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def charToArray(target):
    array = []
    for char in target:
        target_array = np.zeros(26)
        target_array[ord(char[0]) - 97] = 1
        array.append(target_array)
    return np.array(array)


def arrayToChar(array_target):
    array = []
    for a in array_target:
        array.append(chr(a+97))
    return array


def train(sess, train_data, train_target, opt):
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        n_batch = int(train_data.shape[0] / 50)
        for i in range(n_batch):
            batch_x = train_data[i * 100:(i + 1) * 100]
            batch_y = train_target[i * 100:(i + 1) * 100]
            sess.run(opt, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})


def cross_validation(sess, learning_rate, train_data, train_target):
    accuracy_list = []
    kf = KFold(n_splits=5)
    sess.run(tf.global_variables_initializer())
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    i = 0
    for i_t, i_v in kf.split(train_data, train_target):
        print("\t\tLearning rate = " + str(learning_rate) + " N. fold = " + str(i))
        train_input = train_data[i_t]
        train_output = train_target[i_t]

        validating_input = train_data[i_v]
        validating_output = train_target[i_v]

        train(sess, train_input, train_output, opt)
        acc = sess.run(accuracy, feed_dict={x: validating_input, y: validating_output, keep_prob: 1.0})
        accuracy_list.append(acc)
        i += 1
    return np.mean(accuracy_list)


def test(sess, test_data, test_target):
    return sess.run([accuracy, predicted_y], feed_dict={x: test_data, y: test_target, keep_prob: 1.0})


print("Point 1 - Deep Network")
train_data = np.array(pd.read_csv("train-data.csv", header=None))
train_target = np.array(pd.read_csv("train-target.csv", header=None))
test_data = np.array(pd.read_csv("test-data.csv", header=None))
test_target = np.array(pd.read_csv("test-target.csv", header=None))
print("\tData loaded")


x = tf.placeholder(tf.float32, [None, 128])
y = tf.placeholder(tf.float32, [None, 26])
x_image = tf.reshape(x, [-1, 16, 8, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

predicted_y = tf.argmax(y_hat, 1)
real_y = tf.argmax(y, 1)
correct_prediction = tf.equal(predicted_y, real_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------------------------

sess = tf.InteractiveSession()

print("\tData target converting")
array_train_target = charToArray(train_target)
array_test_target = charToArray(test_target)

print("Point 2 - Cross Validation")
learning_rate = 0.00001
accuracy_scores = []
iterations = 5
for i in range(iterations):
    print("\trunning " + str(i) + "/" + str(iterations))
    acc = cross_validation(sess, learning_rate, train_data, array_train_target)
    accuracy_scores.append(acc)
    print("\t\tAccuracy = " + str(acc))
    learning_rate += 0.00001

report = open("kfold.txt", 'w')
report.write(str(accuracy_scores) + '\n')
report.write(str(np.mean(accuracy_scores)))
report.close()

print("Point 3 - Train")
best_index = np.array(accuracy_scores).argmax()
best_learning_rate = 0.00001 + (best_index * 0.00001)
print("\tBest learning rate selected = " + str(best_learning_rate))
opt = tf.train.AdamOptimizer(learning_rate=best_learning_rate).minimize(cross_entropy)
train(sess, train_data, array_train_target, opt)

print("Point 4 - Test")
accuracy, array_test_hat = test(sess, test_data, array_test_target)

print("\tData target converting")
test_hat = arrayToChar(array_test_hat)

print("\tSave accuracy")
f = open("test.txt", 'w')
f.write(str(accuracy))
f.close()

print("Point 5 - Save labels")
f = open("labels.txt", 'w')
for out in test_hat:
    f.write(out + "\n")
f.close()
