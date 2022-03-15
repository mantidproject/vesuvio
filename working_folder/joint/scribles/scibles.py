import numpy as np
import matplotlib.pyplot as plt

# fig, (ax1, ax2)  = plt.subplots(1, 2)
# ax1.plot([1, 3, 5], color='blue', label="blue")
# ax1.plot([2, 4, 6], color='red', label="red")

# handle, label = ax1.get_legend_handles_labels()
# print(handle, label)
# fig.legend(handle, label)

# plt.show()

r = 5
zero = 3
n = 30
x = np.linspace(-r**(1/3), r**(1/3), n)
y = x**3

print(x[:5])
print(y[:5])
