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
