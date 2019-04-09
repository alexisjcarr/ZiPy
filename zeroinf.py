import numpy as np
import scipy as sp 
from scipy.special import logit, expit

class Zeroinf:
    def _init_(self,formula, data, dist=’poisson’, link=’logit’):
        ## call and formula
        '''
        cl <- match.call()
        if(missing(data)) data <- environment(formula)
        mf <- match.call(expand.dots = FALSE)
        m <- match(c("formula", "data", "subset", "na.action", "weights", "offset"), names(mf), 0)
        mf <- mf[c(1, m)]
        mf$drop.unused.levels <- TRUE
        '''

        ## extended formula processing
        '''
        if(length(formula[[3]]) > 1 && identical(formula[[3]][[1]], as.name("|")))
        {
            ff <- formula
            formula[[3]][1] <- call("+")
            mf$formula <- formula
            ffc <- . ~ .
            ffz <- ~ .
            ffc[[2]] <- ff[[2]]
            ffc[[3]] <- ff[[3]][[2]]
            ffz[[3]] <- ff[[3]][[3]]
            ffz[[2]] <- NULL
        } else {
            ffz <- ffc <- ff <- formula
            ffz[[2]] <- NULL
        }
        if(inherits(try(terms(ffz), silent = TRUE), "try-error")) {
            ffz <- eval(parse(text = sprintf( paste("%s -", deparse(ffc[[2]])), deparse(ffz) )))
        }

        '''
        ## call model.frame()
        '''
        mf[[1]] <- as.name("model.frame")
        mf <- eval(mf, parent.frame())
        '''

        ## extract terms, model matrices, response
        self.mt
        self.mtX
        self.X
        
        self.mtZ
        self.Z

        self.Y

        ## convenience variables
        self.n = len(Y)
        self.kx = self.X.shape[1] #number of columns in X matrix
        self.kz = self.Z.shape[1] #number of columns in Z matrix
    	self.Y0 = self.Y <= 0        	
        self.Y1 = self.Y > 0

    def ziPoisson(self, parms): 
		'''
		Log-likelihood for zeroinfl
		
        aside: parms a function that gets or sets simulation model parameters, 
        main or sub-equations, initial values, time steps or solvers and 
        extract simulation results.
        '''
		## count mean
        mu = np.exp(self.X @ parms[1:kx] + offsetx) #offsets will be set in constructor
		## binary mean
        phi = expit(Z @ parms[(kx+1):(kx+kz)] + offsetz))
        # vector of linkinv(Z @ parms[(kx+1):(kx+kz)] + offsetz))
        '''
        For now: Just hard code the inverse of logit
        link <- ## whether it's logit etc.
        linkstr <- match.arg(link)
            matches string argument to name in list
            is there a link list in R code? yes.
            it's in function def 
            maybe don't need rn TODO
        linkobj <- make.link(linkstr)
            turns linkstr into link object
        linkinv <- linkobj$linkinv
            extracts inverse link function from link object
        '''
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log( phi + np.exp( log(1-phi) – mu ) ) 
        loglik1 = np.log(1-phi) + sp.stats.poisson.pmf(Y, lambda=mu)
        ## collect and return
        loglik = sum(loglik[Y0] + loglik[Y1])
        ##loglik <- sum(weights[Y0] * loglik0[Y0]) + sum(weights[Y1] * loglik1[Y1])
        return loglik # return loglik
        '''
        Above is the dot product, I believe. So you are taking 
        Σ[weights[Y0] • (loglik0[Y0] + Σ(weights[Y1] • loglik1[Y1])]
        weights = 1, so yeah. 
        '''

    def gradPoisson(self, parms):
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
			wres_count = -np.exp(-np.log(dens0) + log(1-muz) + clogdens0 + log(mu)))

		if(Y1):
			wres_zero = -1/(1-muz)* #(linkobk$mu.eta(etaz))
		else:
			wres_zero = #(linkobj$mu.eta(etaz) – np.exp(clogdens0) * linkobj$mu.eta(etaz))/dens0) # looking for def for that

		return # column sums of these two columns bound
			# 1) wres_count * weights * X
			# 2) wres_zero * weights * Z
