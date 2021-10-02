
#testing a jupyter notebook

#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
np.set_printoptions(suppress=True, precision=4, linewidth=150)

newPath = r"C:/Users/guijo/Desktop/optimizations/scaling_parameters_improved.npz"
oldPath = r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz"

newResults = np.load(newPath)
oldResults = np.load(oldPath)

newPars = newResults["all_spec_best_par_chi_nit"][0]
oldPars = oldResults["all_spec_best_par_chi_nit"][0]


totalMask = np.isclose(newPars, oldPars, rtol=0.00001, equal_nan = True)
plt.figure(0)
plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
plt.title("Comparison orginal par and new par")
plt.xlabel("pars")
plt.ylabel("spectrums")
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

syntheticFitPath = r"C:/Users/guijo/Desktop/optimizations/scaling_parameters_improved.npz"
syn = np.load(syntheticFitPath)

ws = syn["all_fit_workspaces"][0, :, :-1]
ws = np.where(ws==0, np.nan, ws)
ncp = syn["all_tot_ncp"][0]
par = syn["all_spec_best_par_chi_nit"][0]

meanChi2 = np.nanmean(par[:, -2])
print("Mean Chi2 for spectrums is: ", meanChi2)
print("specNo18: ", ncp[15, :5])

x = np.linspace(0, 1, len(ncp[0]))
plt.figure(3)
spec_idx = 0
plt.plot(x, ws[spec_idx], label="synthetic ncp", linewidth = 2)
plt.plot(x, ncp[spec_idx], "--", label="fitted ncp", linewidth = 2)
plt.legend()
plt.show()

ncp_mask = np.isclose(ws, ncp, rtol=0.01, equal_nan = True)
plt.figure(0)
plt.imshow(ncp_mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
plt.title("Comparison between ws and ncp")
plt.xlabel("TOF")
plt.ylabel("spectrums")
plt.show()

# %%
class someObj:
    def __init__(self):
        self.attribute = 3

def main():

    a = [5]

    def func():
        a[0] += 1
        print(a)
    func()
main()

# %%

class someObj:
    def __init__(self):
        self.no = 3

def main():

    a = someObj()

    def func():
        a.no += 1
        print(a.no)
    func()
main()

#%%

class someObj:
    name = "some_name"
    city = name + "_andACity"
    def __init__(self):
        self.no = 3
        self.nameAndNo = str(self.no) + self.name 
    

def main():

    a = someObj()

    def func():
        a.no += 1
        print(a.no)
        print(a.name)
        print(a.nameAndNo)
        print(a.city)
    func()
main()