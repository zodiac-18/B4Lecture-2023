import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)

fig, ax = plt.subplots()

im = ax.imshow(data)

#cbar = ax.figure.colorbar(im, ax=ax)

#cbar.ax.set_ylabel('Intensity', rotation=-90, va="bottom")

plt.show()