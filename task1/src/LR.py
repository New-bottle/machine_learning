import tensorflow as tf
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(1)
tf.set_random_seed(1)
# hyper parameters
alpha_val = 0.10
learning_rate = 0.01
training_rounds = 200


with open('../data/test_vec.pk1', 'rb') as f:
	x_vals = pickle.load(f)
	y_vals = pickle.load(f)
#iris = 
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

feature_num = x_vals.shape[1]

# partition
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace = False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# model and loss function's definition
batch_size = 100
x_data = tf.placeholder(shape=[None, feature_num], dtype = tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype = tf.float32)

W = tf.Variable(tf.random_normal(shape=[feature_num, 1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

pred = tf.matmul(x_data, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=pred, logits=y_target))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_target,1))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_vec = []
for i in range(training_rounds):
	rand_index = np.random.choice(len(x_vals_train), size = batch_size)
	rand_x = x_vals_train[rand_index]
	rand_y = np.transpose([y_vals_train[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(cost, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	if (i + 1) % (training_rounds / 10) == 0:
		print ('training :', (i + 1.0) / (training_rounds / 10) * 10, '%',)
		print('Loss = ' + str(temp_loss))

plt.plot(loss_vec)
plt.legend(['损失'])
plt.ylim(0.,1.)
plt.show()



