import matplotlib.pyplot as plt
import numpy as np
  
# create data
x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y1 = [3.177, 0.5854, 0.4469, 0.3825, 0.3718, 0.2977, 0.3007, 0.2830, 0.2658, 0.2518, 0.2162, 0.2909, 0.2573, 0.2496, 0.2732, 0.2382, 0.2176, 0.1850, 0.1946, 0.2194, 0.2177, 0.2026, 0.2461, 0.1703, 0.1854]

x2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y2 = [5.1310, 2.1717, 2.6287, 1.5747, 1.1132, 0.8087, 1.6818, 0.7773, 0.8688, 0.7153, 1.3899, 0.5254, 0.6237, 0.6093, 1.0534, 1.1534, 0.4642, 0.4921, 1.2331, 0.2394, 0.4395, 0.7649, 0.6401, 0.4195, 0.2365]  
plt.plot(x1, y1, label = "Loss", linestyle="-", marker='o')
plt.plot(x2, y2, label = "Validation Loss", linestyle="-", marker='o')
plt.yticks([0.1,1.0,2.0,3.0,4.0,5.0])
plt.title('Accuracy over 25 epochs')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
#plt.grid(True)
plt.grid(True,which='major', color='#DDDDDD')
plt.grid(True, which='minor', color='#EEEEEE', linewidth = 0.5)
plt.minorticks_on()
plt.legend()
plt.show()
