import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2)  = plt.subplots(1, 2)
ax1.plot([1, 3, 5], color='blue', label="blue")
ax1.plot([2, 4, 6], color='red', label="red")

handle, label = ax1.get_legend_handles_labels()
print(handle, label)
fig.legend(handle, label)

plt.show()