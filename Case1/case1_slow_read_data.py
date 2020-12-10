
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
np.set_printoptions(threshold = np.inf)

uwb_sample_rate = 6.5
imu_sample_rate = 100.0

imu_path = "./case1_slow_IMU_split.txt"
uwb_path = "./case1_slow_UWB.txt"

# uwb data
import_uwb = np.loadtxt(uwb_path)
print(import_uwb.shape)

uwb_t = import_uwb[:,0]
uwb_data = import_uwb[:,1]
uwb_num_point = uwb_t.shape[0]
uwb_total_time = uwb_t[-1]
print(uwb_total_time)


# groudtruth 
true_dis = []
true_time = [8.15, 14.42, 22.57, 29.36, 36.31, 41.03, 47.99, 54.08, 61.18, 67.45, 74.52]
len_split = len(true_time)
true_time_point = np.zeros(len_split)
true_time_duration = np.zeros(len_split)
close_distance = 0.9
far_distance = 4.5

for uwb_time_i in range(0,uwb_num_point-1):

	for time_i in range(len_split):
		if uwb_t[uwb_time_i] < true_time[time_i] and uwb_t[uwb_time_i+1] > true_time[time_i]:
			true_time_point[time_i] = uwb_time_i

print(true_time_point)

true_time_duration[0] = true_time_point[0]

for time_i in range(1,len_split):
	true_time_duration[time_i] = true_time_point[time_i] - true_time_point[time_i-1]

print(true_time_duration)

for time_i in range(len_split):


	process_indicator = time_i % 4
	if process_indicator == 0:
		speed = (far_distance - close_distance) / (true_time_duration[time_i])
		down = np.arange(far_distance, close_distance, -speed)
		true_dis.extend(list(down))
	elif process_indicator == 1:
		down_flat = close_distance * np.ones(int(true_time_duration[time_i]))
		true_dis.extend(list(down_flat))
	elif process_indicator == 2:
		speed = (far_distance - close_distance) / (true_time_duration[time_i])
		up = np.arange(close_distance, far_distance, speed)
		true_dis.extend(list(up))
	else:
		up_flat = far_distance * np.ones(int(true_time_duration[time_i]))
		true_dis.extend(list(up_flat))

# print(true_dis)
if len(uwb_data) > len(true_dis):
	data_len = len(true_dis)
else:
	data_len = len(uwb_data)

# imu data
import_imu = np.loadtxt(imu_path)
print(import_imu.shape)

imu_num_point = int(uwb_t[data_len]*imu_sample_rate) + 1
imu_data = import_imu[0:imu_num_point,:]
imu_t = np.arange(0, imu_num_point) / imu_sample_rate
print(imu_t[-1])

uwb_save = np.zeros((data_len, 3))
uwb_save[:,0] = uwb_t[0:data_len]
uwb_save[:,1] = uwb_data[0:data_len]
uwb_save[:,2] = true_dis[0:data_len]
np.savetxt("./case1_slow_align/case1_slow_uwb_align.txt", uwb_save)


imu_save = np.zeros((imu_num_point, 10))
imu_save[:,0] = imu_t[0:imu_num_point]
imu_save[:,1:] = imu_data[0:imu_num_point]
np.savetxt("./case1_slow_align/case1_slow_imu_align.txt", imu_save)


# plt.figure()


# plt.subplot(3,1,1)
# plt.plot(uwb_t[0:data_len],uwb_data[0:data_len], label='UWB_measure')
# plt.plot(uwb_t[0:data_len], true_dis[0:data_len], label='groudtruth')
# plt.title("Case 1 slow")
# plt.ylabel('Distance (m)')
# plt.legend(loc='right', fontsize='x-small')


# plt.subplot(3,1,2)
# plt.plot(uwb_t[0:data_len],abs(uwb_data[0:data_len] - true_dis[0:data_len]))
# plt.ylabel('UWB Error (m)')
# plt.ylim(0,1)



# plt.subplot(3,1,3)
# # plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 0:3])
# # plt.ylabel('IMU Acc')
# # plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 3:6])
# # plt.ylabel('IMU Gyro')
# plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 6:9])
# plt.ylabel('IMU Mag')
# plt.xlabel('Time (s)')
# plt.show()
