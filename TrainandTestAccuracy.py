import matplotlib.pyplot as plt
import numpy as np
  
# create data
x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y1 = [0.7351, 0.8214, 0.8337, 0.8522, 0.8789, 0.8932, 0.8850, 0.8830, 0.8953, 0.8953, 0.9158, 0.8994, 0.8953, 0.9076, 0.8994, 0.9097, 0.9158, 0.9405, 0.9384, 0.9189, 0.9097, 0.9261, 0.935, 0.9343, 0.9322 ]

x2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
y2 = [0.2523, 0.4595, 0.4775, 0.7477, 0.5766, 0.6847, 0.5405, 0.6306, 0.7928, 0.7568, 0.6847, 0.8739, 0.8649, 0.8559, 0.8378, 0.8378, 0.8739, 0.8378, 0.7297, 0.9055, 0.8829, 0.8559, 0.8649, 0.8198, 0.8929]  
plt.plot(x1, y1, label = "Train_acc", linestyle="-", marker='o')
plt.plot(x2, y2, label = "Test_acc", linestyle="-", marker='o')
plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
plt.title('Accuracy over 25 epochs')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
#plt.grid(True)
plt.grid(True,which='major', color='#DDDDDD')
plt.grid(True, which='minor', color='#EEEEEE', linewidth = 0.5)
plt.minorticks_on()
plt.legend()
plt.show()
