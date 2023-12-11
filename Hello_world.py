import numpy as np
import matplotlib.pyplot as plt 

theta = np.arange(0,2*np.pi,0.1)
y = np.sin(theta)

plt.scatter(theta,y)
plt.show()
