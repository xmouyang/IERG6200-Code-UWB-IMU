
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
# predict_uwb_path = "./case2_slow_post_uwb_1.txt"

imu_path = "./case2_fast_imu_align.txt"
uwb_path = "./case2_fast_uwb_align.txt"
predict_uwb_path = "./case2_fast_post_uwb_1.txt"


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


uwb_t, uwb_data, true_dis, imu_t, imu_data = import_data(uwb_path, imu_path)
predic_uwb_data = np.loadtxt(predict_uwb_path)
original_uwb_error = abs(uwb_data - true_dis)
predict_uwb_error = abs(predic_uwb_data - true_dis)

data_len = uwb_t.shape[0]
train_point = int(data_len*0.7) + 1


plt.figure()


plt.subplot(3,1,1)

plt.plot(uwb_t[0:train_point],predic_uwb_data[0:train_point], label='UWB_train')
plt.plot(uwb_t[train_point:],uwb_data[train_point:], 'r', label='measured_distance')
plt.plot(uwb_t[train_point:],predic_uwb_data[train_point:], 'g', label='DNN_calibrated_distance')
plt.plot(uwb_t[0:data_len], true_dis[0:data_len], label='groudtruth')
# plt.title("Case 2 Slow")
plt.title("Case 2 Fast")
plt.legend(loc='upper left', fontsize='x-small')
plt.ylabel('Distance (m)')


plt.subplot(3,1,2)
plt.plot(uwb_t[0:train_point],predict_uwb_error[0:train_point],label='UWB_train_error')
plt.plot(uwb_t[train_point:],original_uwb_error[train_point:], 'r', label='measured_error')
plt.plot(uwb_t[train_point:],predict_uwb_error[train_point:], 'g', label='DNN_calibrated_error')
plt.legend(loc='upper left', fontsize='x-small')
plt.ylim(0,1)
plt.ylabel('UWB Error (m)')

plt.subplot(3,1,3)
# plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 0:3])
# plt.ylabel('IMU Acc')
plt.plot(imu_t,imu_data[:,3:6])
plt.ylabel('IMU Gyro')
# plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 6:9])
# plt.ylabel('IMU Mag')
# plt.xlabel('Time (s)')
plt.show()
