import numpy as np
import matplotlib.pyplot as plt

#x = np.linspace(0, 10, 1000)
x = np.arange(0.5, 5, 0.5)
y = np.exp(-x)
z = y + 0.5

plt.figure(figsize=(8,4))
plt.plot(x,y,"b--",label="$NMF$")
plt.plot(x,z,label="$JPPCF$",color="red",linewidth=2)
plt.xlabel("Timestep")
plt.ylabel("recall")
plt.title("NMF & JPPCF RMSE")
plt.legend()
plt.show()
