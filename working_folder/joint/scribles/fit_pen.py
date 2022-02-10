# Trying out fitting with penalization
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Build some noisy data

def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


def HermitePolynomialPen1(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
    penalization = np.abs(np.sum(func[func<0])) * 1
    return func + penalization


def HermitePolynomialPen2(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
    penalization = np.abs(np.sum(func[func<0])) * 10
    return func + penalization


x = np.linspace(-20, 20, 100)
dataY = HermitePolynomial(x, 1, 0, 5, 1, -3) + np.random.random(x.size)*0.01

p0 = [1, 0, 4, 0, 0]
popt1, pcov1 = optimize.curve_fit(HermitePolynomialPen1, x, dataY, p0=p0)
popt2, pcov2 = optimize.curve_fit(HermitePolynomialPen2, x, dataY, p0=p0)

yfit1 = HermitePolynomial(x, *popt1)
yfit2 = HermitePolynomial(x, *popt2)

# Try it with minimize()

def residuals(p0, x, dataY):
    fun = HermitePolynomial(x, *p0)
    return np.sum((dataY - fun)**2)


def constr(p0):
    x0, sigma1, c4, c6 = p0[1:]
    return (1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))

constraints = ({'type': 'ineq', 
                'fun': constr})

result = optimize.minimize(residuals, p0, args=(x, dataY), method="SLSQP", constraints=constraints)
popt3 = result["x"]
yfit3 = HermitePolynomial(x, *popt3)

degreesFreedom = x.size - len(p0) 
chi1 = residuals(popt1, x, dataY) / degreesFreedom
chi2 = residuals(popt2, x, dataY) / degreesFreedom
chi3 = residuals(popt3, x, dataY) / degreesFreedom

print("chi2 1: ", chi1)
print("chi2 2: ", chi2)
print("chi2 3: ", chi3)
print(popt1)
print(popt2)
print(popt3)

# Hessian = result["hess"]
# invHessian = result["hess_inv"]
# print(invHessian)

# Main issue is how to compute covariance matrix from minimize()
# pcov3 = result["hess_inv"]
# print("covariance matrices: ", pcov1, pcov3)


plt.plot(x, dataY, "o")
plt.fill_between(x, yfit1, color="red", alpha=0.4, label="scale=1")
plt.fill_between(x, yfit2, color="green", alpha=0.4, label="scale=10")
plt.fill_between(x, yfit3, color="yellow", alpha=0.4, label="minimize")
plt.legend()
plt.show()

