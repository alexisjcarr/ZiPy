#Convenience variables
n = len(Y)
kx= X.shape[1]  #NCOL(X)
kz= Z.shape[1]  #NCOL(Z)
if Y>0:    #Y0 <- Y <= 0
    Y1=Y
else:      #Y1 <- Y > 0
    Y0=Y

#We only code for one option to deal with missing values and that is to drop the rows that have nan at any column 

data = data.dropna(axis=0) #drop rows with missing value

#Libraries needed so far
import numpy as np
import scipy as sp

#Maximum Likelihood function

def ziPoisson(parms):
        mu= np.vectorize(exp(X @ parms[1:kx]) + offsetx)    #mean. kx is no of columns of X.
        phi= np.vectorize(*linkinv*(Z @ parms[(kx+1):(kx+kz)] + offsetz)) # need to find linkinv function in python or alternative
        return sum(*dzipois*(Y, lambda = mu, pi = phi, log = TRUE)) * weights) # find zero inflated poisson distribution function in python
        
        
#using scipy.optimize to maximize the function 
from scipy.optimize import minimize
f = -------  # function to be MAXIMIZED.
res = minimize(lambda x: -f(x), 0)

## for detailed options for optimization
 from scipy.optimize import OptimizeResult
def custmin(f, x0, args=(), maxfev=None, stepsize=0.1,
...         maxiter=100, callback=None, **options):


