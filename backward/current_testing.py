
#testing a jupyter notebook

#%%
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=150)

newResults = np.load(r"C:/Users/guijo/Desktop/optimizations/scaling_parameters_improved.npz")
oldResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz")

newPars = newResults["all_spec_best_par_chi_nit"][0]
oldPars = oldResults["all_spec_best_par_chi_nit"][0]
print(newPars[:3])
print(oldPars[:3])


totalMask = np.isclose(newPars, oldPars, rtol=0.01, equal_nan = True)
plt.figure(0)
plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
plt.title("Comparison orginal par and new par")
plt.xlabel("pars")
plt.ylabel("spectrums")
plt.show()


# %%
import numpy as np
newResults = np.load(r"C:/Users/guijo/Desktop/optimizations/scaling_parameters_improved.npz")
oldResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz")

#np.testing.assert_allclose(newResults["all_mean_intensities"], oldResults["all_mean_intensities"])

for key in oldResults:
    try:
        print("\nevaluating: ",key)
        np.testing.assert_allclose(newResults[key][0], oldResults[key][0], rtol=1e-2)            
        print("shape: ", newResults[key].shape)
    except KeyError:
        print("KeyError: one of the results doesnt have this key")
    # except AssertionError:
    #     print("Assertion Error")

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

syn = np.load(r"C:/Users/guijo/Desktop/optimizations/scaling_parameters_improved.npz")
#np.load(r"C:\Users\guijo\Desktop\work_repos\scatt_scripts\backward\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_synthetic.npz")

ws = syn["all_fit_workspaces"][0, :, :-1]
ncp = syn["all_tot_ncp"][0]
par = syn["all_spec_best_par_chi_nit"][0]

meanChi2 = np.nanmean(par[:, -2])
print("Mean Chi2 for spectrums is: ", meanChi2)

x = np.linspace(0, 1, len(ncp[0]))
plt.figure(3)
spec_idx = 0
# print("first values ws: ", ws[spec_idx, :5])
# print("first values ncp: ", ncp[spec_idx, :5])
plt.plot(x, ws[spec_idx], label="synthetic ncp", linewidth = 2)
plt.plot(x, ncp[spec_idx], "--", label="fitted ncp", linewidth = 2)
plt.legend()
plt.show()

ncp_mask = np.isclose(ws, ncp, rtol=0.001, equal_nan = True)
plt.figure(0)
plt.imshow(ncp_mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
plt.title("Comparison between ws and ncp")
plt.xlabel("TOF")
plt.ylabel("spectrums")
plt.show()

# %%
