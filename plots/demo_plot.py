import numpy as np
import matplotlib.pyplot as plt
import random

#x = np.linspace(0, 10, 1000)
x = range(2, 11)
bpmf = [0.3, 0.35, 0.4, 0.5, 0.65, 0.7, 0.76, 0.87, 0.9]
timeSVDpp = [0.2, 0.28, 0.34, 0.38, 0.45, 0.56, 0.67, 0.71, 0.84]
WALS = [0.25, 0.30, 0.37, 0.40, 0.48, 0.65, 0.70, 0.75, 0.86]
TensorALS = [0.28, 0.38, 0.44, 0.48, 0.55, 0.58, 0.7, 0.78, 0.88]
BTMF = [0.29, 0.40, 0.46, 0.50, 0.58, 0.60, 0.72, 0.82, 0.90]
TRM = [0.30, 0.43, 0.48, 0.54, 0.59, 0.62, 0.75, 0.83, 0.91]
TTARM = [0.33, 0.47, 0.50, 0.56, 0.60, 0.65, 0.77, 0.84, 0.92]

plt.figure(figsize=(10, 8))

plt.plot(x, bpmf, "b*--", label="$BPMF$")
plt.plot(x, timeSVDpp, "d--", label="$timeSVD++$", color='seagreen')
plt.plot(x, WALS, "y^--", label="$WALS$")
plt.plot(x, TensorALS, "kh--", label="$TensorALS$")
plt.plot(x, BTMF, "cp--", label="$BTMF$")
plt.plot(x, TRM, "rs-", label="$TRM$")
plt.plot(x, TTARM, "m8-", label="$TTARM$")

plt.xlabel("Timestep")
#plt.ylabel("Recall@300")
plt.xlim(2, 10)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, 0.1))
# plt.title("models performance on Recall@300")

plt.legend(loc='upper left', numpoints=1)
plt.show()
