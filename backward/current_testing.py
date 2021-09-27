
#testing a jupyter notebook

#%%
import numpy as np
import scipy

arrayA = np.arange(12)
shapeOfA = (3,4)
arrayA = arrayA.reshape(shapeOfA)
print(arrayA)

selected = arrayA.flatten() * np.ones((3,1)) [False, False, True]
print(selected)

#%%
import sys
for p in sys.path:
    print(p)

#%%
%load_ext snakeviz
%snakeviz import Final_optimized_bckwd_script

# %%
import numpy as np
newResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_cleanest.npz")
oldResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz")

#np.testing.assert_allclose(newResults["all_mean_intensities"], oldResults["all_mean_intensities"])

for key in oldResults:
    try:
        print("\nevaluating: ",key)
        np.testing.assert_allclose(newResults[key][0], oldResults[key][0], rtol=1e-4)            
        print("shape: ", newResults[key].shape)
    except KeyError:
        print("KeyError: one of the results doesnt have this key")
    except AssertionError:
        print("Assertion Error")

# %%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

syn = np.load(r"C:/Users/guijo/Desktop/optimizations/scipy_optimization_test.npz")
#np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_synthetic.npz")

ws = syn["all_fit_workspaces"][0, :, :-1]
ncp = syn["all_tot_ncp"][0]

x = np.linspace(0, 1, len(ncp[0]))
plt.figure(3)
spec_idx = 5
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
# Comparing the three different approaches of optimization

scalePars = np.load(r"C:/Users/guijo/Desktop/optimizations/scaling_parameters.npz")
partLSQ = np.load(r"C:/Users/guijo/Desktop/optimizations/partitioned_least_squares.npz")
scipyOpt = np.load(r"C:/Users/guijo/Desktop/optimizations/scipy_optimization.npz")

print(partLSQ["all_spec_best_par_chi_nit"].shape)
for key in scalePars:
    print(key)

for file in [scalePars, partLSQ, scipyOpt]:
    pars = file["all_spec_best_par_chi_nit"].reshape(132, 15)
    chi2 = pars[:, -2]
    print(np.nanmax(chi2))
    maxMask = chi2 == np.nanmax(chi2)
    print("nit max:", pars[:, -1][maxMask])
    specMax = np.argwhere(maxMask)
    print("specMax: ", specMax)
    meanChi2 = np.nansum(chi2)
    print("The mean Chi2 is ", meanChi2)
    widths = file["all_mean_widths"]
    intensities = file["all_mean_intensities"]
    print(widths, "\n", intensities)

#%%
# Investigating the spectrums that are badly fit

scalePars = np.load(r"C:/Users/guijo/Desktop/optimizations/scaling_parameters.npz")
partLSQ = np.load(r"C:/Users/guijo/Desktop/optimizations/partitioned_least_squares.npz")
scipyOpt = np.load(r"C:/Users/guijo/Desktop/optimizations/scipy_optimization.npz")

for file in [scalePars, scipyOpt]:
    ws = file["all_fit_workspaces"][0, :, :-1]
    ncp = file["all_tot_ncp"][0, :]

    x = np.linspace(0, 1, len(ncp[0]))
    plt.figure()
    spec_idx = 15
    plt.plot(x, ws[spec_idx], label="synthetic ncp", linewidth = 2)
    plt.plot(x, ncp[spec_idx], "--", label="fitted ncp", linewidth = 2)
    plt.legend()
    plt.show()

    ncp_mask = np.isclose(ws, ncp, rtol=0.01, equal_nan = True)
    plt.figure()
    plt.imshow(ncp_mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
    plt.title("Comparison between synthetic and fitted ncp")
    plt.xlabel("TOF")
    plt.ylabel("spectra")
    plt.show()

    # Take out spectra that have a bad fit
    badFitMask = ncp_mask[:, 20] # At this bin there isnt noise
    print("No of good spec: ", badFitMask.sum())
    ws = ws[badFitMask, :]
    ncp = ncp[badFitMask, :]
    print("one of the masked spectrum:\n", ws[15, :5])


    ncp_mask = np.isclose(ws, ncp, rtol=0.01, equal_nan = True)
    plt.figure()
    plt.imshow(ncp_mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
    plt.title("Comparison between synthetic and fitted ncp")
    plt.xlabel("TOF")
    plt.ylabel("spectra")
    plt.show()   

    pars = file["all_spec_best_par_chi_nit"][0]
    print("Bad spectrums: ", pars[~badFitMask, 0])
    goodPars = pars[badFitMask, :]
    totalMean = np.nanmean(goodPars, axis=0)
    print("All means:\n", totalMean)





# %%
