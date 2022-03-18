from global_final import main 
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

load = False
nSpec = 45
nGroupsRange = np.linspace(1, nSpec, int(nSpec/2)).astype(int)   # Zero in last position means nSpec==nGroups
print(nGroupsRange)

if not load:

    allValues = np.zeros((len(nGroupsRange), 3))
    allErrors = np.zeros(allValues.shape)
    for i, nGroups in enumerate(nGroupsRange):
        try:
            values, errors = main(nSpec, nGroups, False)
            sharedValues = values[:3]
            sharedErrors = errors[:3]
        except AssertionError:
            sharedValues = np.zeros(3)
            sharedErrors = np.zeros(3)

        allValues[i] = sharedValues
        allErrors[i] = sharedErrors

    for vals, errs, p in zip(allValues.T, allErrors.T, ["sigma1", "c4", "c6"]):
        plt.errorbar(nGroupsRange, vals, errs, fmt=".", label=p)

    np.savez("./global_fit_analysis.npz", allValues=allValues, allErros=allErrors)

else:
    data = np.load("./global_fit_analysis.npz")
    allValues = data["allValues"]
    allErrors = data["allErros"]

    for vals, errs, p in zip(allValues.T, allErrors.T, ["sigma1", "c4", "c6"]):
        plt.errorbar(nGroupsRange, vals, errs, fmt="o", label=p)

print("\nvalues sigma1, c4 and c6:\n", allValues)


plt.title(f"Number of spectra: {nSpec}")
plt.xlabel("Number of groups")
plt.ylabel("Values")
plt.legend()
plt.show()