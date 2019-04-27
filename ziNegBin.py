def ziNegBin(self,parms)
        '''
        Log likelihood for Negative Binomial Model
        '''
        ## count mean
        mu = np.exp(self.X @ parms[np.arange(self.kx)] + self.offsetx) 
        ## binary mean
        phi = expit(self.Z @ parms[np.arange((self.kx),(self.kx+self.kz))] + self.offsetz)
        # expit is inverse link of logit
        #theta is measure of overdispersion with respect to the Poisson distribution
        theta = np.exp(parms[np.arrange(self.kx + self.kz)])
        prob = theta/(theta+mu)
        #nbinom only supports parameters of p and size(theta in our case)
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log( phi + np.exp( np.log(1-phi) + sp.stats.nbinom.pmf(0, theta, prob) ) )  
        loglik1 = np.log(1-phi) + sp.stats.nbinom.pmf(self.Y,theta,prob)
       
        ## collect and return
        loglik = self.weights[self.Y0] @ loglik0[self.Y0] + self.weights[self.Y1]@loglik1[self.Y1] 

        return loglik

#  gradNegBin <- function(parms) {
#     ## count mean
#     eta <- as.vector(X %*% parms[1:kx] + offset)
#     mu <- exp(eta)
#     ## binary mean
#     etaz <- as.vector(Z %*% parms[(kx+1):(kx+kz)])
#     muz <- linkinv(etaz)
#     ## negbin size
#     theta <- exp(parms[(kx+kz)+1])

#     ## densities at 0
#     clogdens0 = sp.stats.nbinom.pmf(0, theta, prob)
#     dens0 <- muz * (1 - as.numeric(Y1)) + exp(log(1 - muz) + clogdens0)

#     ## working residuals  
#     wres_count <- ifelse(Y1, Y - mu * (Y + theta)/(mu + theta), -exp(-log(dens0) +
#       log(1 - muz) + clogdens0 + log(theta) - log(mu + theta) + log(mu)))
#     wres_zero <- ifelse(Y1, -1/(1-muz) * linkobj$mu.eta(etaz),
#       (linkobj$mu.eta(etaz) - exp(clogdens0) * linkobj$mu.eta(etaz))/dens0)
#     wres_theta <- theta * ifelse(Y1, digamma(Y + theta) - digamma(theta) +
#       log(theta) - log(mu + theta) + 1 - (Y + theta)/(mu + theta),
#       exp(-log(dens0) + log(1 - muz) + clogdens0) *
#       (log(theta) - log(mu + theta) + 1 - theta/(mu + theta)))

#     colSums(cbind(wres_count * weights * X, wres_zero * weights * Z, wres_theta))
  


def gradNegBin(self, parms):
        '''
        Gradient function for NegBin Model
        '''
        ## count mean
        eta = self.X @ parms[np.arange(self.kx)] + self.offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = self.Z @ parms[np.arange((self.kx),(self.kx+self.kz))] + self.offsetz
        muz = expit(etaz) 
        ## negbin size
        theta = np.exp(parms[np.arrange(self.kx + self.kz)])
        prob = theta/(theta+mu)
        ## densities at 0
        clogdens0 = sp.stats.nbinom.pmf(0, theta, prob)
        dens0 = muz * (1 - self.Y1.astype(float)) + np.exp(np.log(1-muz) + clogdens0 + np.log(mu))
        
        ## mu_eta = d(mu)/d(eta); derivative of inverse link function
        mu_eta = np.exp(etaz)/(1 + np.exp(etaz))**2
        ## working residuals  
       
        wres_count = np.where(self.Y1, self.Y - mu * ((self.Y + theta)/ (mu + theta)), -np.exp(-np.log(dens0) + np.log(1-muz) + clogdens0
                                                                        + np.log(theta) - np.log(mu + theta) + np.log(mu)))
        wres_zero = np.where(self.Y1, -1/(1-muz) * mu_eta, mu_eta - np.exp(clogdens0) * (mu_eta)/dens0) 
        
        wres_theta = theta * np.where(self.Y1, sp.special.digamma(self.Y + theta) - sp.special.digamma(theta) + np.log(theta) 
                                      - np.log(mu + theta) + 1 - (self.Y + theta)/(mu+theta), np.exp(-np.log(dens0) + np.log(1-muz) + clogdens0)
                                      * (np.log(theta) - np.log(mu + theta) + 1 - theta/(mu+theta)))
        ######################
        return np.hstack((np.expand_dims(wres_count*self.weights,axis=1)*self.X,
                    np.expand_dims(wres_zero*self.weights,axis=1)*self.Z), np.expand_dims(wres_theta,axis=1).sum(axis=0))
        
        


    
