import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
# from sklearn.model_selection import train_test_split
np.set_printoptions(threshold = np.inf)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras import regularizers

# imu_path = "./case1_slow_imu_align.txt"
# uwb_path = "./case1_slow_uwb_align.txt"

imu_path = "./case1_fast_imu_align.txt"
uwb_path = "./case1_fast_uwb_align.txt"

NUM_OF_CLASS = 3

def import_data(uwb_path, imu_path):

	# uwb data
	import_uwb = np.loadtxt(uwb_path)
	print(import_uwb.shape)

	uwb_t = import_uwb[:,0]
	uwb_data = import_uwb[:,1]
	true_dis = import_uwb[:,2]

	uwb_num_point = uwb_t.shape[0]
	uwb_total_time = uwb_t[-1]
	print(uwb_total_time)

	# imu data
	import_imu = np.loadtxt(imu_path)
	print(import_imu.shape)

	imu_t = import_imu[:,0]
	imu_data = import_imu[:,1:]
	print(imu_t[-1])

	return uwb_t, uwb_data, true_dis, imu_t, imu_data


def data_pre(uwb_t, imu_t, imu_data, true_time, train_time_point):

	uwb_num_point = uwb_t.shape[0]
	seconds = 3
	data_len = uwb_num_point - seconds
	imu_feature_len = int(17*9*seconds)

	x_feature = np.zeros((data_len, imu_feature_len))
	y_label = np.zeros(data_len)
	train_point = 0

	for time_i in range(seconds,uwb_num_point):
		imu_start_point = int(uwb_t[time_i-seconds] * 100)
		imu_end_point = imu_start_point + 17*seconds
		imu_spiece_data = imu_data[imu_start_point:imu_end_point]
		x_feature[time_i-seconds] = (imu_spiece_data).reshape(-1)

	for point_i in range(data_len):

		time_i = uwb_t[point_i]

		for negtive_i in range(0,19,4):
			if time_i > true_time[negtive_i] and time_i < true_time[negtive_i+1]:
				y_label[point_i] = 2
		for positive_i in range(2,19,4):
			if time_i > true_time[positive_i] and time_i < true_time[positive_i+1]:
				y_label[point_i] = 1

		if time_i < true_time[train_time_point] and time_i + 1 > true_time[train_time_point]:
			train_point = point_i
			print(train_point)

	y_label = y_label.astype(int)

	print(x_feature.shape)
	print(y_label.shape)

	x_train = x_feature[0:train_point]
	x_test = x_feature[train_point:]
	y_train = y_label[0:train_point]
	y_test = y_label[train_point:]

	return x_train,x_test,y_train,y_test


 
def local_run(x_train,x_test,y_train,y_test, local_rho, local_learning_rate, local_iter):

  # 4 layers
  model = keras.Sequential([
  
  keras.Input(shape=(459,)),
  layers.Dense(300,activation = 'relu', kernel_regularizer=regularizers.l2(local_rho)),
  layers.Dense(150,activation = 'relu', kernel_regularizer=regularizers.l2(local_rho)),
  layers.Dense(80,activation = 'relu', kernel_regularizer=regularizers.l2(local_rho)),
  layers.Dense(40,activation = 'relu', kernel_regularizer=regularizers.l2(local_rho)),
  layers.Dense(10,activation = 'relu', kernel_regularizer=regularizers.l2(local_rho)),
  layers.Dense(NUM_OF_CLASS,activation = 'softmax', kernel_regularizer=regularizers.l2(local_rho) ),
  
  ])


  # model train
  sgd = tf.keras.optimizers.SGD(learning_rate=local_learning_rate)
  model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
  history = model.fit(x_train, y_train, batch_size = 16, epochs = local_iter, verbose=1)
  print(K.eval(model.optimizer.lr))
  print("total run epoch:",len(history.history['loss']))

  # model evaluate
  score = model.evaluate(x_test, y_test, verbose=0)
  node_accuracy = score[1]
  y_prediction = np.argmax(model.predict(x_test), axis=-1)

  return node_accuracy, y_prediction


uwb_t, uwb_data, true_dis, imu_t, imu_data = import_data(uwb_path, imu_path)
true_time = [0, 3.6, 6.64, 10.34, 13.8, 17.12, 21.24, 24.72, 28.19, 32.15, 35.58, 39.45, 44.15, 47.81, 52.93, 56.75, 61.25, 65.13, 69.87, 73.38]
train_time_point = 12

x_train,x_test,y_train,y_test = data_pre(uwb_t, imu_t, imu_data, true_time, train_time_point)
print(x_train.shape)
print(y_train.shape)
print(y_train)
print(y_test)
y_class_train = to_categorical(y_train, NUM_OF_CLASS)
y_class_test = to_categorical(y_test, NUM_OF_CLASS)


local_rho = 1e-2 # 1e-2 for fast
local_learning_rate = 1e-3
local_iter = 4000
accuracy, y_prediction = local_run(x_train,x_test,y_class_train,y_class_test, local_rho, local_learning_rate, local_iter)
print(accuracy)
print(y_prediction)
print(y_test)

for i in range(y_prediction.shape[0]):
	if y_prediction[i] == 2:
		y_prediction[i] = -1
	if y_test[i] == 2:
		y_test[i] = -1

print(y_prediction)
print(y_test)

plt.figure()
plt.plot(y_prediction, label='y_prediction')
plt.plot(y_test, label='y_test')
plt.title("Case 1 Fast - Velocity Prediction")
plt.legend(loc='lower right', fontsize='x-small')
plt.show()
np.savetxt("case1_fast_predict_velocity.txt", y_prediction)
plt.ylim(-1,1)

#0.84765625




