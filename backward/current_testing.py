
#testing a jupyter notebook

#%%
print("exciting new worl!")

# %%
import numpy as np
newResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild_synthetic.npz")
oldResults = np.load(r".\script_runs\opt_spec3-134_iter4_ncp_nightlybuild.npz")

np.testing.assert_allclose(newResults["all_mean_intensities"][0], oldResults["all_mean_intensities"][0])

for key in oldResults:
    try:
        print("evaluating: ",key)
        np.testing.assert_allclose(newResults[key][0], oldResults[key][0], rtol=1e-4)            
        print(newResults[key].shape)
    except KeyError:
        pass

# %%
