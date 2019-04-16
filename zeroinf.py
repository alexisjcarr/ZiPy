import numpy as np
import scipy as sp 
import pandas as pd
from scipy.special import logit, expit
import patsy as pat
import statsmodels.api as sm

class Zeroinf:
    def __init__(self, formula, data, dist='poisson', link='logit'):
        '''
        Class Constructor
        '''
        self.formula = formula
        self.data = data
        self.call = 'Zeroinf(formula = '+ self.formula + ', data = df, dist = \'poisson\', link = \'logit\')'
        
        ## da matrices
        self.X, self.Y, self.Z = self.formula_extraction(self.formula, self.data)
        
        ## convenience variables
        self.n = len(self.Y)
        self.kx = self.X.shape[1] #number of columns in X matrix
        self.kz = self.Z.shape[1] #number of columns in Z matrix
        self.Y0 = self.Y <= 0  
        self.Y1 = self.Y > 0
        self.theta = None
        self.log_theta = None
        
        ## offsets and weights
        self.offsetz = np.repeat(0, self.n)
        self.offsetx = np.repeat(0, self.n)
        self.weights = np.repeat(1, self.n)
        
    @staticmethod
    def formula_extraction(formula, data):
        X_, Z_0 = formula.split('|')
        Z_ = '{}~{}'.format(formula.split('~')[0],Z_0)
        Y, X = pat.dmatrices(X_, data, return_type='dataframe')
        Y = np.squeeze(Y)
        Z = pat.dmatrices(Z_, data, return_type='dataframe')[1]

        return X, Y, Z
    
    def ziPoisson(parms): # parms supplied by self.mlEstimation.
        '''
        Log-likelihood for ZIP Model
        '''
        ## count mean
        mu = np.exp(X @ parms[np.arange(kx)] + offsetx) 
        ## binary mean
        phi = expit(Z @ parms[np.arange((kx),(kx+kz))] + offsetz)
        # expit is inverse link of logit
        
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) ) 
        loglik1 = np.log(1-phi) + sp.stats.poisson.pmf(Y, mu)
        ## collect and return
        loglik = weights[Y0] @ loglik0[Y0] + weights[Y1]@loglik1[Y1] 

        return loglik

    def gradPoisson(parms):
        '''
        Gradient function for ZIP Model
        '''
        ## count mean
        eta = X @ parms[np.arange(kx)] + offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = Z @ parms[np.arange((kx),(kx+kz))] + offsetz
        muz = expit(etaz) 

        ## densities at 0
        clogdens0 = -mu
        dens0 = muz * (1 - Y1.astype(float)) + np.exp(np.log(1-muz) + clogdens0)

        ## mu_eta = d(mu)/d(eta); derivative of inverse link function
        mu_eta = np.exp(etaz)/(1 + np.exp(etaz))**2
        
        ## working residuals
        wres_count = np.where(Y1, Y - mu, -np.exp(-np.log(dens0) + np.log(1-muz) + clogdens0))
        wres_zero = np.where(Y1, -1/(1-muz) * mu_eta, mu_eta - np.exp(clogdens0) * (mu_eta)/dens0)

        return np.hstack((np.expand_dims(wres_count*weights,axis=1)*X, \
                    np.expand_dims(wres_zero*weights,axis=1)*Z)).sum(axis=0)

    def startingValues(self):
        '''
        Returns count model and binomial model from glm
        '''
        modelCount = sm.GLM(endog = self.Y, exog = self.X, \
            family = sm.genmod.families.family.Poisson(),\
                weights = self.weights, offset = self.offsetx).fit()
        selfmodelZero = sm.GLM(endog = self.Y0.astype(int), exog = self.Z,\
            family = sm.genmod.families.family.Binomial(link=sm.genmod.families.links.logit), weights = self.weights,\
                offset = self.offsetz).fit()
        
        self.startValues =  {'zeroStartValues': modelZero.params,\
                             'countStartValues': modelCount.params}

        return self.startValues

    def mlEstimation(self):
        fun = self.ziPoisson
        x0 = np.hstack((self.startValues['countStartValues'].values,\
            self.startValues['zeroStartValues'].values))
        method = 'BFGS' #(per paper)
        jac = self.gradPoisson
        options = {'maxiter': 10000, 'disp': False}

        self.fit_ = sp.optimize.minimize(fun = fun, x0 = x0, method = method, jac = jac,\
            options = options) #returns object OptimizeResult

        ## coefficients and covariances
        coefc_keys = []
        for key in self.X.columns.values:
            coefc_keys.append(key)
        coefc_values = []
        for value in self.fit_.x[0:kx]:
            coefc_values.append(value)
            
        coefc = dict(zip(coefc_keys, coefc_values))

        coefz_keys = []
        coefz_values = []
        for key in self.Z.columns.values:
            coefz_keys.append(key)
        for value in self.fit_.x[kx:kx+kz]:
            coefz_values.append(value)
            
        coefz = dict(zip(coefz_keys, coefz_values))

        ## fitted and residuals TODO

        ## effective observations TODO

        print(coefc)
    
    def __repr__(self):
        out = 'Call: \n'
        out += self.call
        out += '\n\nPearson residuals: \n'
        out += 'Min      1Q      Median      3Q      Max\n'
        out += '<min, etc>\n'
        out += 'Count model coefficients (poisson with log link):\n'
        out += '            Estimate Std. | Error | z-value | Pr(>|z|)\n\n'
        out += f'{self.coefc.keys()}\n {self.coefc.values()}\n'
        out += 'Number of iterations in BFGS optimization: <extract this>\n'
        out += 'Log-likelihood: <extract this too>'
        
        return out

        
df = pd.read_csv('~/Downloads/bioChemists.csv')
formula_ = 'art ~ fem + mar + phd + kid5 + ment | 1'
Zeroinf(data=df, formula=formula_)