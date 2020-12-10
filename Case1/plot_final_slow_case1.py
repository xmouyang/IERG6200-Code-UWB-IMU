
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


imu_path = "./case1_slow_predict_velocity.txt"
uwb_path = "./measure_slow.txt"
predict_uwb_path = "./predict_slow.txt"
groudtruth_uwb_path = './ground_truth_slow.txt'

uwb_raw_path = './case1_slow_uwb_align.txt'

true_time = [0, 8.15, 14.42, 22.57, 29.36, 36.31, 41.03, 47.99, 54.08, 61.18, 67.45, 74.52]
train_time_point = 4

import_uwb = np.loadtxt(uwb_raw_path)
print(import_uwb.shape)
uwb_t = import_uwb[:,0]

def get_velocity_gt(uwb_t):

	uwb_num_point = uwb_t.shape[0]
	seconds = 3
	data_len = uwb_num_point - seconds
	y_label = np.zeros(data_len)

	for point_i in range(data_len):

		time_i = uwb_t[point_i]

		if time_i > true_time[0] and time_i < true_time[1]:
			y_label[point_i] = -1
		elif time_i > true_time[2] and time_i < true_time[3]:
			y_label[point_i] = 1
		elif time_i > true_time[4] and time_i < true_time[5]:
			y_label[point_i] = -1
		elif time_i > true_time[6] and time_i < true_time[7]:
			y_label[point_i] = 1
		elif time_i > true_time[8] and time_i < true_time[9]:
			y_label[point_i] = -1
		elif time_i > true_time[10] and time_i < true_time[11]:
			y_label[point_i] = 1

		if time_i < true_time[train_time_point] and time_i + 1 > true_time[train_time_point]:
			train_point = point_i
			print(train_point)

	y_label = y_label.astype(int)
	y_test = y_label[train_point:]

	return y_test, train_point


gt_velocity, train_point = get_velocity_gt(uwb_t)
predict_velocity = np.loadtxt(imu_path)
print(gt_velocity)
print(predict_velocity)
print(gt_velocity.shape)
print(predict_velocity.shape)

groudtruth_uwb_data = np.loadtxt(groudtruth_uwb_path)
measure_uwb_data = np.loadtxt(uwb_path)
predict_uwb_data = np.loadtxt(predict_uwb_path)
print(groudtruth_uwb_data.shape)
print(uwb_t[train_point:].shape)

plt.figure()


plt.subplot(3,1,1)
plt.plot(uwb_t[train_point+4:],gt_velocity[1:], label='groundtruth_velocity')
plt.plot(uwb_t[train_point+4:],predict_velocity[0:253], label='predict_velocity')
plt.title("Case 1 Slow - Calibration")
plt.legend(loc='upper left', fontsize='x-small')
plt.ylabel('Velocity Prediction')


plt.subplot(3,1,2)
plt.plot(uwb_t[train_point+1:],groudtruth_uwb_data, label='groudtruth_distance')
plt.plot(uwb_t[train_point+1:],measure_uwb_data, label='measured_distance')
plt.plot(uwb_t[train_point+1:],predict_uwb_data, label='kalman_calibrated_distance')
plt.legend(loc='upper left', fontsize='x-small')
plt.ylabel('Distance (m)')

plt.subplot(3,1,3)
plt.plot(uwb_t[train_point+1:],abs(measure_uwb_data - groudtruth_uwb_data), label='measured_error')
plt.plot(uwb_t[train_point+1:],abs(predict_uwb_data - groudtruth_uwb_data), label='kalman_calibrated_error')
plt.legend(loc='upper left', fontsize='x-small')
plt.ylim(0,1)
plt.ylabel('Error (m)')
plt.show()
