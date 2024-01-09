import numpy as np
import matplotlib.pyplot as plt 

theta = np.arange(0,2*np.pi,0.1)
y = np.sin(theta)

plt.plot(theta,y)
plt.xlabel("t /frame")
plt.tick_params(bottom=False, left=False, right=False, top=False)
plt.show()

