
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold = np.inf)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras import backend as K
from tensorflow.keras import regularizers

imu_sample_rate = 100.0

# imu_path = "./case2_slow_imu_align.txt"
# uwb_path = "./case2_slow_uwb_align.txt"

imu_path = "./case2_fast_imu_align.txt"
uwb_path = "./case2_fast_uwb_align.txt"

NUM_OF_CLASS = 20

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


def data_pre(uwb_t, uwb_data, true_dis, imu_t, imu_data):

	uwb_num_point = uwb_t.shape[0]
	data_len = uwb_num_point - 1
	imu_feature_len = int(17*9)
	feature_len = 2 + imu_feature_len # distance at t and t-1, imu 9-axis * 17

	x_feature = np.zeros((data_len, feature_len))
	for time_i in range(1,uwb_num_point):
		x_feature[time_i-1, 0] = uwb_data[time_i-1]
		x_feature[time_i-1, 1] = uwb_data[time_i]
		imu_start_point = int(uwb_t[time_i-1] * 100)
		imu_end_point = imu_start_point + 17
		imu_spiece_data = imu_data[imu_start_point:imu_end_point]
		x_feature[time_i-1, 2:] = (imu_spiece_data).reshape(-1)

	distance_error = (uwb_data - true_dis + 1) * 10
	print(distance_error)
	error_class = np.round(distance_error)
	print(error_class)
	y_label = error_class[1:]
	y_label = y_label.astype(int)

	print(x_feature.shape)
	print(y_label.shape)
	
	train_point = int(data_len*0.7)

	x_train = x_feature[0:train_point]
	x_test = x_feature[train_point:]
	y_train = y_label[0:train_point]
	y_test = y_label[train_point:]
	y_test_real_error = distance_error[train_point+1:] - 10

	return x_train,x_test,y_train,y_test, y_test_real_error


def local_run(x_train,x_test,y_train,y_test, local_rho, local_learning_rate, local_iter):

  # 4 layers
  model = keras.Sequential([
  
  keras.Input(shape=(155,)),
  layers.Dense(100,activation = 'relu'),
  layers.Dense(80,activation = 'relu'),
  layers.Dense(40,activation = 'relu'),
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
x_train,x_test,y_train,y_test, y_test_real_error = data_pre(uwb_t, uwb_data, true_dis, imu_t, imu_data)
y_class_train = to_categorical(y_train, NUM_OF_CLASS)
y_class_test = to_categorical(y_test, NUM_OF_CLASS)
# print(y_train.shape)


local_rho = 1e-3 # 1e-2 for fast
local_learning_rate = 1e-3
local_iter = 1000
accuracy, y_prediction = local_run(x_train,x_test,y_class_train,y_class_test, local_rho, local_learning_rate, local_iter)
print(accuracy)
print(y_prediction)
print(y_test)

y_test_real_error = y_test_real_error / 10.0
test_prediction_dis_error = (y_prediction - 10)/10.0
post_dis_error = y_test_real_error - test_prediction_dis_error
print(y_test_real_error)
print(test_prediction_dis_error)
print(post_dis_error)


real_mean_error = np.mean(abs(y_test_real_error))
predict_mean_error = np.mean(abs(post_dis_error))
print(real_mean_error)
print(predict_mean_error)

train_len = y_train.shape[0]
post_uwb_data = np.zeros(uwb_data.shape[0])
post_uwb_data[0:train_len+1] = uwb_data[0:train_len+1] 
post_uwb_data[train_len+1:] = post_dis_error + true_dis[train_len+1:]
# print(post_uwb_data)
# np.savetxt("case2_slow_post_uwb_1.txt", post_uwb_data)
np.savetxt("case2_fast_post_uwb_1.txt", post_uwb_data)



plt.figure()
plt.subplot(3,1,1)
plt.plot(uwb_t[train_len+1:], y_test_real_error, 'r', label='measured_error')
plt.plot(uwb_t[train_len+1:], test_prediction_dis_error, 'g', label='predict_error')
# plt.title("Case 2 Slow - Calibration ")
plt.title("Case 2 Fast - Calibration ")
plt.legend(loc='lower right', fontsize='x-small')
plt.ylim(-1,1)

plt.subplot(3,1,2)
plt.plot(uwb_t[train_len+1:], y_test_real_error+4.5, label='measured_distance')
plt.plot(uwb_t[train_len+1:], true_dis[train_len+1:], label='groundtruth_distance')
plt.plot(uwb_t[train_len+1:], post_dis_error+4.5, 'g', label='DNN_calibrated_distance')
plt.legend(loc='lower right', fontsize='x-small')
# plt.ylim(-1,1)



plt.subplot(3,1,3)
plt.plot(uwb_t[train_len+1:], abs(y_test_real_error), 'r', label='measured_error')
plt.plot(uwb_t[train_len+1:], abs(post_dis_error), 'g', label='DNN_calibrated_error')
plt.legend(loc='upper right', fontsize='x-small')
plt.ylim(0,1)
plt.show()