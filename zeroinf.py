import numpy as np
import scipy as sp 
from scipy.special import logit, expit
import patsy as pat

class Zeroinf:
    def _init_(self,formula, data, dist=’poisson’, link=’logit’):
        '''
        ***Constructor***
        '''
        '''
        The following variables are defined within formula_extraction

        self.X # count data model regressors
        self.Z # zero-inflated model regressors
        self.Y 
        '''
        self.formula = formula
        self.data = data
        self.offsetz, self.offsetx = 0.0
        self.weights = 1.0

        ## convenience variables
        self.n = len(self.Y)
        self.kx = self.X.shape[1] #number of columns in X matrix
        self.kz = self.Z.shape[1] #number of columns in Z matrix
    	self.Y0 = self.Y <= 0        	
        self.Y1 = self.Y > 0

    def formula_extraction(self):
        self.X_, self.Z_ = formula.str.split('|')
        self.Z_ = '{beg}{z_}'.format(beg=formula.str.split('~')[0],z_=Z_)
        self.Y_, self.X = pat.dmatrices(X_, data, return_type='dataframe')
        self.Z = pat.dmatrices(Z_, data, return_type='dataframe')[1]
        self.Y = np.squeeze(self.Y_)
        

    def ziPoisson(self, parms): 
		'''
		***Log-likelihood for Zeroinf***
        '''
		## count mean
        mu = np.exp(self.X @ parms[1:self.kx] + self.offsetx) 
		## binary mean
        phi = expit(self.Z @ parms[(self.kx+1):(self.kx+self.kz)] + self.offsetz))
        # expit is inverse link of logit
       
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log( phi + np.exp( log(1-phi) – mu ) ) 
        loglik1 = np.log(1-phi) + sp.stats.poisson.pmf(Y, lambda=mu)
        ## collect and return
        loglik = sum(self.weights[self.Y0] @ loglik0[self.Y0]) + \
            sum(self.weights[self.Y1] @ loglik1[self.Y1])
        return loglik

    def gradPoisson(self, parms):
        '''
        ***Gradient likelihood for Zeroinf***
        '''
		## count mean
		eta = # X • parms[1:kx} + offsetx
		mu = np.exp(eta)
		## binary mean
		etaz = # z • parms[{kx+1):(kx+kz)] + offsetz)
		muz = # linkinv(etaz) # will probably need Jones help

		## densities at 0
		clogdens0 = -mu
		dens0 = muz * (1 – Y1) + np.exp(np.log(1-muz) + clogdens0)

		## working residuals
		if(Y1):
			wres_count = Y – mu
		else:
			wres_count = -np.exp(-np.log(dens0) + log(1-muz) +\ 
                clogdens0 + log(mu)))

		if(Y1):
			wres_zero = -1/(1-muz)* #(linkobk$mu.eta(etaz))
		else:
			wres_zero = #(linkobj$mu.eta(etaz) – np.exp(clogdens0) * linkobj$mu.eta(etaz))/dens0) # looking for def for that

		return # column sums of these two columns bound
			# 1) wres_count * weights * X
			# 2) wres_zero * weights * Z

    def __repr__(self):
        '''
        part of the print function will go here
        '''
