import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.arange(20)

f = np.sin(x)
f[:3] = np.nan
f[8:10] = np.nan
interp = interpolate.interp1d(x, f)

xp = np.linspace(0, 18.5, 20)
# yp = np.interp(xp, x, f)
yp = interp(xp)

print(yp)
plt.plot(x, f, label="Raw")
plt.plot(xp, yp, "x", label="Interp")
plt.show()
