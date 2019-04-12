import numpy as np
import scipy as sp 
import pandas as pd
from scipy.special import logit, expit
import patsy as pat
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Binomial

class Zeroinf:
    def __init__(self, formula, data, dist='poisson', link='logit'):
        '''
        Class Constructor
        '''
        self.formula = formula
        self.data = data
        
        ## da matrices
        self.X, self.Y, self.Z = self.formula_extraction(self.formula, self.data)
        
        ## convenience variables
        self.n = len(self.Y)
        self.kx = self.X.shape[1] #number of columns in X matrix
        self.kz = self.Z.shape[1] #number of columns in Z matrix
        self.Y0 = self.Y <= 0  
        self.Y1 = self.Y > 0
        self.theta = NULL
        self.log_theta = NULL
        
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
    
    def ziPoisson(self, parms): # parms supplied by self.mlEstimation.
        '''
        Log-likelihood for ZIP Model
        '''
        ## count mean
        mu = np.exp(self.X @ parms[1:self.kx] + self.offsetx) 
        ## binary mean
        phi = expit(self.Z @ parms[(self.kx+1):(self.kx+self.kz)] + self.offsetz)
        # expit is inverse link of logit
       
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log( phi + np.exp( log(1-phi) - mu ) ) 
        loglik1 = np.log(1-phi) + sp.stats.poisson.pmf(self.Y, mu)
        ## collect and return
        loglik = sum(self.weights[self.Y0] @ loglik0[self.Y0]) + \
            sum(self.weights[self.Y1] @ loglik1[self.Y1]) #weights need to be matrices

        return loglik

    def gradPoisson(self, parms):
        '''
        Gradient function for ZIP Model
        '''
        ## count mean
        eta = self.X @ parms[1:kx] + self.offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = self.Z @ parms[(kx+1):(kx+kz)] + offsetz
        muz = expit(etaz) 

        ## densities at 0
        clogdens0 = -mu
        dens0 = muz @ (1 – self.Y1.astype(float)) + np.exp(np.log(1-muz) + clogdens0)

        ## mu_eta = d(mu)/d(eta); derivative of inverse link function
        mu_eta = np.exp(etaz)/(1 + np.exp(etaz))**2

        ## working residuals
        if(Y1):
            wres_count = self.Y – mu
        else:
            wres_count = -np.exp(-np.log(dens0) + log(1-muz) + clogdens0 + exp.log(mu)))

        if(Y1):
            wres_zero = -1/(1-muz) * mu_eta 
        else:
            wres_zero = mu_eta - np.exp(clogdens0) * mu_eta/dens0 

        return np.hstack((wres_count @ self.weights @ self.X),\ 
                         (wres_zero @ self.weights @ self.Z)) # likely incorrect, fix
        # column sums of these two columns bound
            # 1) wres_count * weights * X
            # 2) wres_zero * weights * Z

    def startingValues(self):
        '''
        Returns count model and binomial model from glm
        '''
        modelCount = sm.GLM(endog = self.Y, exog = self.X, family = Poisson(),\
             weights = self.weights, offset = self.offsetx).fit()
        modelZero = sm.GLM(endog = self.Y0.astype(int), exog = self.Z,\
            family = Binomial(link=logit), weights = self.weights,\
                 offset = self.offsetz).fit()

        self.startValues =  {'zeroStartValues': modelZero.params,\
            'countStartValues': modelCount.params}

        return self.startValues

    def mlEstimation(self):
        fun = self.ziPoisson
        x0 = self.startValues # TODO (extracted)
        method = 'BFGS' #(per paper)
        jac = self.gradPoisson
        options = {'maxiter': 10000, 'disp': True}

        self.fit = sp.optimize.minimize(fun = fun, x0 = x0, method = method, jac = jac,\
             options = options) #returns object OptimizeResult

        ## coefficients and covariances
        '''
        coefc <- fit$par[1:kx]
        names(coefc) <- names(start$count) <- colnames(X)
        coefz <- fit$par[(kx+1):(kx+kz)]
        names(coefz) <- names(start$zero) <- colnames(Z)

        colnames(vc) <- rownames(vc) <- c(paste("count", colnames(X), sep = "_"), \
            paste("zero",  colnames(Z), sep = "_"))
        '''

        ## fitted and residuals
        '''
        mu <- exp(X %*% coefc + offsetx)[,1]
        phi <- linkinv(Z %*% coefz + offsetz)[,1]
        Yhat <- (1-phi) * mu
        res <- sqrt(weights) * (Y - Yhat)
        '''

        ## effective observations
        '''
        nobs <- sum(weights > 0) ## = n - sum(weights == 0)

        rval <- list(coefficients = list(count = coefc, zero = coefz),
            residuals = res,
            fitted.values = Yhat,
            optim = fit,
            method = method,
            control = ocontrol,
            start = start,
            weights = if(identical(as.vector(weights), rep.int(1L, n))) NULL else weights,
            offset = list(count = if(identical(offsetx, rep.int(0, n))) NULL else offsetx,
            zero = if(identical(offsetz, rep.int(0, n))) NULL else offsetz),
            n = nobs,
            df.null = nobs - 2,
            df.residual = nobs - (kx + kz + (dist == "negbin")),
            terms = list(count = mtX, zero = mtZ, full = mt),
            theta = theta,
            SE.logtheta = SE.logtheta,
            loglik = fit$value,
            vcov = vc,
            dist = dist,
            link = linkstr,
            linkinv = linkinv,
            converged = fit$convergence < 1,
            call = cl,
            formula = ff,
            levels = .getXlevels(mt, mf),
            contrasts = list(count = attr(X, "contrasts"), zero = attr(Z, "contrasts"))
        )
        if(model) rval$model <- mf
        if(y) rval$y <- Y
        if(x) rval$x <- list(count = X, zero = Z)
            
        class(rval) <- "zeroinfl"
        return(rval)
        }
        '''
        return {} # return the hell out of that dictionary

    def __repr__(self):
        '''
        Change as needed for testing
        '''
