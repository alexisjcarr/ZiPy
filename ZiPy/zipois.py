import numpy as np
import scipy as sp
import pandas as pd
from scipy.special import logit, expit
import patsy as pat
import statsmodels.api as sm
import statsmodels as sfm
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

class zipois:
    def __init__(self, formula, data, link='logit'):
        '''
        Class Constructor
        '''
        self.formula = formula
        self.data = data
        self.call = 'zipois(formula = ' + self.formula + \
            ', data = df, link = \'logit\')'

        # da matrices
        self.X, self.Y, self.Z = self.formula_extraction( \
            self.formula, self.data)

        # convenience variables
        self.n = len(self.Y)
        self.kx = self.X.shape[1]  # number of columns in X matrix
        self.kz = self.Z.shape[1]  # number of columns in Z matrix
        self.Y0 = self.Y <= 0
        self.Y1 = self.Y > 0
        self.theta = None
        self.log_theta = None

        ## offsets and weights
        self.offsetz = np.repeat(0, self.n)
        self.offsetx = np.repeat(0, self.n)
        self.weights = np.repeat(1, self.n)

        self.modelZero, self.modelCount = self.startingValues( \
            self.X, self.Y, self.Z, self.weights, self.offsetx, self.Y0, self.offsetz)
        self.modelParams = {'zeroStartValues': self.modelZero.params, \
                            'countStartValues': self.modelCount.params}

        self.coefc_keys, self.coefc_values, self.coefz_keys,\
            self.coefz_values = \
            self.mlEstimation(x0=np.hstack((self.modelParams['countStartValues'].values,
                                            self.modelParams['zeroStartValues'].values)))

    @staticmethod
    def formula_extraction(formula, data):
        X_, Z_0 = formula.split('|')
        Z_ = '{}~{}'.format(formula.split('~')[0], Z_0)
        Y, X = pat.dmatrices(X_, data, return_type='dataframe')
        Y = np.squeeze(Y)
        Z = pat.dmatrices(Z_, data, return_type='dataframe')[1]

        return X, Y, Z

    # parms supplied by self.mlEstimation.
    def ziPoisson(self, parms, sign=1.0):
        '''
        Log-likelihood for ZIP Model
        '''
        # count mean
        mu = np.exp(self.X @ parms[np.arange(self.kx)]) + self.offsetx

        # binary mean
        phi = expit(self.Z @ parms[np.arange( \
            (self.kx), (self.kx+self.kz))] + self.offsetz)
        # expit is inverse link of logit

        # log-likelihood for y = 0 and y >= 1
        loglik0 = np.log(phi + np.exp(np.log(1-phi) - mu))
        loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(self.Y, mu)
        # collect and return
        loglik = self.weights[self.Y0] @ loglik0[self.Y0] + \
            self.weights[self.Y1] @ loglik1[self.Y1]

        return sign*loglik

    def gradPoisson(self, parms, sign=1.0):
        '''
        Gradient function for ZIP Model
        '''
        # count mean
        eta = self.X @ parms[np.arange(self.kx)] + self.offsetx
        mu = np.exp(eta)
        # binary mean
        etaz = self.Z @ parms[np.arange(
            (self.kx), (self.kx+self.kz))] + self.offsetz
        muz = expit(etaz)

        # densities at 0
        clogdens0 = -mu
        dens0 = muz * (1 - self.Y1.astype(float)) + \
            np.exp(np.log(1-muz) + clogdens0)

        # mu_eta = d(mu)/d(eta); derivative of inverse link function
        mu_eta = (np.exp(-etaz)/(1 + np.exp(etaz))**2)

        # working residuals
        wres_count = np.where(self.Y1, self.Y - mu,\
                              -np.exp(-np.log(dens0) + np.log(1-muz) + clogdens0 + np.log(mu)))
        wres_zero = np.where(self.Y1, -1/(1-muz) * mu_eta, (mu_eta - \
                                                            np.exp(clogdens0) * mu_eta)/dens0)

        return sign*(np.hstack((np.expand_dims(wres_count*self.weights, axis=1)*self.X,\
                                np.expand_dims(wres_zero*self.weights, axis=1)*self.Z)).sum(axis=0))

    # have this called in the constructor and return those variables
    def startingValues(self, X, Y, Z, weights, offsetx, Y0, offsetz):
        '''
        Returns count model and binomial model from glm
        '''
        modelCount = sm.GLM(endog=self.Y, exog=self.X,\
                            family=sm.genmod.families.Poisson(\
                                link=sm.genmod.families.links.log),\
                            freq_weights=self.weights, offset=self.offsetx).fit()
        modelZero = sm.GLM(endog=self.Y0.astype(int), exog=self.Z,\
                           family=sm.genmod.families.Binomial(link=sm.genmod.families.links.logit), freq_weights=self.weights,\
                           offset=self.offsetz).fit()

        # self.startValues =  {'zeroStartValues': modelZero.params,\
        #                    'countStartValues': modelCount.params}

        return modelZero, modelCount

    def mlEstimation(self, x0):  # have this called in the constructor and return those variables
        fun = self.ziPoisson
        jac = self.gradPoisson
        method = 'Nelder-Mead'
        options = {'maxiter': 10000, 'disp': True}

        fit_ = sp.optimize.minimize(fun, x0=x0, method=method, jac=jac,\
                                    options=options, args=(-1.0,))  # returns object OptimizeResult

        ## coefficients and covariances
        coefc_keys = []
        coefc_values = []
        for key in self.X.columns.values:
            coefc_keys.append(key)
        for value in fit_.x[0:self.kx]:
            coefc_values.append(value)
#         coefc = dict(zip(coefc_keys, coefc_values))

        coefz_keys = []
        coefz_values = []
        for key in self.Z.columns.values:
            coefz_keys.append(key)
        for value in fit_.x[self.kx:self.kx+self.kz]:
            coefz_values.append(value)
#         coefz = dict(zip(coefz_keys, coefz_values))

        ## fitted and residuals
        # mu = np.exp(np.dot(self.X, coefc_values) + self.offsetx)
        # phi = expit(np.dot(self.Z, coefz_values) + self.offsetz)
        # yhat = (1 - phi) * mu
        # res = np.sqrt(self.weights) * (self.Y - yhat)

        # # effective observations
        # nobs = np.sum(self.weights > 0)

        # Pearson residuals
        # pearson_res = np.round(st.mstats.mquantiles(res,
        #                                             prob=[0, 0.25, 0.5, 0.75, 1.0]), 5)

        return coefc_keys, coefc_values, coefz_keys, coefz_values

    def __repr__(self):
        print('Call:')
        print(self.call + '\n')
#         print('Pearson residuals:')
#         print('Min\t  1Q\t     Median\t3Q\t Max')
#         print(*self.pearson_res, sep=' | ')
#         print('\n')
        print('Count model coefficients (poisson with log link):')
        #out += '            Estimate Std. | Error | z-value | Pr(>|z|)\n\n'

        fmt = '%-20s%s%s'
        print(fmt % ('', '', 'Estimate'))
        for i, (key, value) in enumerate(zip(self.coefc_keys, self.coefc_values)):
            print(fmt % (key, '', value))
        print('\n')

        print('Zero-inflation model coefficients (binomial with logit link):')
        print(fmt % ('', '', 'Estimate'))
        for i, (key, value) in enumerate(zip(self.coefz_keys, self.coefz_values)):
            print(fmt % (key, '', value))

        return ''
