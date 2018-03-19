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
print (x_data.shape)
y_target = tf.placeholder(shape=[None, 1], dtype = tf.float32)

print (x_data.shape)
A = tf.Variable(tf.random_normal(shape=[feature_num, 1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
print (A.shape)

model_output = tf.matmul(x_data, A) + b
print (model_output.shape)

l2_norm = tf.reduce_sum(tf.square(A))

# loss = max(0, 1-pred*acutal) + alpha * L2_norm(A)^2
alpha = tf.constant([alpha_val])
#classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1.,model_output), y_target))
classification_term = tf.reduce_mean(tf.maximum(0., 1.-model_output * y_target))
loss = classification_term + alpha*l2_norm

# output
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
# training
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(training_rounds):
	rand_index = np.random.choice(len(x_vals_train), size = batch_size)
	rand_x = x_vals_train[rand_index]
	rand_y = np.transpose([y_vals_train[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(temp_loss)
	train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
	train_accuracy.append(train_acc_temp)
	test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
	test_accuracy.append(test_acc_temp)
	if (i + 1) % (training_rounds / 10) == 0:
		print ('training :', (i + 1.0) / (training_rounds / 10) * 10, '%',)
		print('Loss = ' + str(test_acc_temp))

plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['损失','训练精确度','测试精确度'])
plt.ylim(0.,1.)
plt.show()
