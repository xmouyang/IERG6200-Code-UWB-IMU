import matplotlib.pyplot as plt
import numpy as np
import math

# file_path = '/Users/xavier/Downloads/uwb_data_seg.txt'
file_path = '/Users/xavier/Desktop/siuman/UWB_kalman/uwb_data_fast.txt'
uwb_data = np.loadtxt(file_path)
indicator_path = '/Users/xavier/Desktop/siuman/UWB_kalman/case1_fast_predict_velocity.txt'
indicator = np.loadtxt(indicator_path)
# true_time = [0, 8.15, 14.42, 22.57, 29.36, 36.31, 41.03, 47.99, 54.08, 61.18, 67.45, 74.52]
true_time = [0, 3.6, 6.64, 10.34, 13.8, 17.12, 21.24, 24.72, 28.19, 32.15, 35.58, 39.45, 44.15, 47.81, 52.93, 56.75, 61.25, 65.13, 69.87, 73.38]

time = uwb_data.T[0]
measured = uwb_data.T[1]
truth = uwb_data.T[2]

t = np.zeros(uwb_data.shape[0])
for i in range(1,uwb_data.shape[0]):
	t[i] = time[i]-time[i-1]

v = 0.5
position = truth
position_noise = measured

import matplotlib.pyplot as plt


# 初试的估计导弹的位置就直接用GPS测量的位置
predicts = [position_noise[0]]
position_predict = predicts[0]

predict_var = 0
odo_var = 683**2 #这是我们自己设定的位置测量仪器的方差，越大则测量值占比越低
v_std = 50 # 测量仪器的方差
for i in range(1,time.shape[0]):
# 	if time[i] > true_time[0] and time[i] < true_time[1]:
# 		dv = -1
# 	elif time[i] > true_time[2] and time[i] < true_time[3]:
# 		dv = 1
# 	elif time[i] > true_time[4] and time[i] < true_time[5]:
# 		dv = -1
# 	elif time[i] > true_time[6] and time[i] < true_time[7]:
# 		dv = 1
# 	elif time[i] > true_time[8] and time[i] < true_time[9]:
# 		dv = -1
# 	elif time[i] > true_time[10] and time[i] < true_time[11]:
# 		dv = 1
# 	elif time[i] > true_time[12] and time[i] < true_time[13]:
# 		dv = -1
# 	elif time[i] > true_time[14] and time[i] < true_time[15]:
# 		dv = 1
# 	elif time[i] > true_time[16] and time[i] < true_time[17]:
# 		dv = -1
# 	elif time[i] > true_time[18] and time[i] < true_time[19]:
# 		dv = 1								
# 	else:
# 		dv = 0
	dv = 1*indicator[i]
	position_predict = position_predict + dv*t[i]
	predict_var += v_std**2 
	
	position_predict = position_predict*odo_var/(predict_var + odo_var)+position_noise[i]*predict_var/(predict_var + odo_var)
	predict_var = (predict_var * odo_var)/(predict_var + odo_var)**2
	predicts.append(position_predict)

print(np.mean(np.abs(predicts-position)))
print(np.mean(np.abs(position_noise-position)))

np.savetxt('ground_truth_fast.txt',position)
np.savetxt('measure_fast.txt',position_noise)
np.savetxt('predict_fast.txt',predicts)

plt.subplot(2,1,1)
plt.plot(time,position,label='truth position')
plt.plot(time,position_noise,label='only use measured position')
plt.plot(time,predicts,label='kalman filtered position')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time,np.abs(position_noise-position),label='only use measured position')
plt.plot(time,np.abs(predicts-position),label='kalman filtered position')

plt.legend()
plt.show()
