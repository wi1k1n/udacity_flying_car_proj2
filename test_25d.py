import numpy as np


TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 5

data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)