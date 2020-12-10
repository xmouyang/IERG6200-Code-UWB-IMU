
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
np.set_printoptions(threshold = np.inf)

imu_sample_rate = 100.0

# imu_path = "./case2_slow_IMU_split.txt"
# uwb_path = "./case2_slow_UWB.txt"

imu_path = "./case2_fast_IMU_split.txt"
uwb_path = "./case2_fast_UWB.txt"


# uwb data
import_uwb = np.loadtxt(uwb_path)
print(import_uwb.shape)

uwb_t = import_uwb[:,0]
uwb_data = import_uwb[:,1]
uwb_num_point = uwb_t.shape[0]
uwb_total_time = uwb_t[-1]
print(uwb_total_time)


# groudtruth 
true_dis = 4.5 * np.ones(uwb_num_point)
data_len = uwb_num_point - 1


# imu data
import_imu = np.loadtxt(imu_path)
print(import_imu.shape)

imu_num_point = int(uwb_t[data_len]*imu_sample_rate) + 1
imu_data = import_imu[0:imu_num_point,:]
imu_t = np.arange(0, imu_num_point) / imu_sample_rate
print(imu_t[-1])


# uwb_save = np.zeros((data_len, 3))
# uwb_save[:,0] = uwb_t[0:data_len]
# uwb_save[:,1] = uwb_data[0:data_len]
# uwb_save[:,2] = true_dis[0:data_len]
# np.savetxt("./case2_slow_align/case2_slow_uwb_align.txt", uwb_save)


# imu_save = np.zeros((imu_num_point, 10))
# imu_save[:,0] = imu_t[0:imu_num_point]
# imu_save[:,1:] = imu_data[0:imu_num_point]
# np.savetxt("./case2_slow_align/case2_slow_imu_align.txt", imu_save)

uwb_save = np.zeros((data_len, 3))
uwb_save[:,0] = uwb_t[0:data_len]
uwb_save[:,1] = uwb_data[0:data_len]
uwb_save[:,2] = true_dis[0:data_len]
np.savetxt("./case2_fast_align/case2_fast_uwb_align.txt", uwb_save)


imu_save = np.zeros((imu_num_point, 10))
imu_save[:,0] = imu_t[0:imu_num_point]
imu_save[:,1:] = imu_data[0:imu_num_point]
np.savetxt("./case2_fast_align/case2_fast_imu_align.txt", imu_save)


# plt.figure()


# plt.subplot(3,1,1)
# plt.plot(uwb_t[0:data_len],uwb_data[0:data_len], label='UWB_measure')
# plt.plot(uwb_t[0:data_len], true_dis[0:data_len], label='groudtruth')
# # plt.title("Case 2 slow")
# plt.title("Case 2 fast")
# plt.legend(loc='upper left', fontsize='x-small')
# plt.ylabel('Distance (m)')


# plt.subplot(3,1,2)
# plt.plot(uwb_t[0:data_len],abs(uwb_data[0:data_len] - true_dis[0:data_len]))
# # plt.xlim(0,30)
# plt.ylim(0,1)
# plt.ylabel('UWB Error (m)')

# plt.subplot(3,1,3)
# # plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 0:3])
# # plt.ylabel('IMU Acc')
# # plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 3:6])
# # plt.ylabel('IMU Gyro')
# plt.plot(imu_t[0:imu_num_point],imu_data[0:imu_num_point, 6:9])
# plt.ylabel('IMU Mag')
# plt.xlabel('Time (s)')
# plt.show()
