import numpy as np
import matplotlib.pyplot as plt

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        pass

# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
        #pass
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        ''' 
        We use the standard Box Muller algorithm to generate a standard normal
        from two independent standard uniforms.
        '''
        #pass
        U1, U2 = np.random.uniform(), np.random.uniform()
        R, theta = np.sqrt(-2*np.log(U1)), 2*np.pi*U2
        Z = R*np.cos(theta)
        return self.mu + Z * self.sigma
    
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        #pass
        self.Mu = Mu
        self.Sigma = Sigma

    def sample(self):
        #pass
        L, Size =  np.linalg.cholesky(self.Sigma), self.Mu.shape
        #We generate d independent standard normals using Box Muller
        U1, U2 = np.random.uniform(size = Size), np.random.uniform(size = Size)
        R, theta = np.sqrt(-2*np.log(U1)), 2*np.pi*U2
        Z = R*np.cos(theta)
        return np.matmul(L,Z)+self.Mu
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
        #pass
        self.ap = ap

    def sample(self):
        #pass
        u = np.random.uniform()
        PartitionUnit = np.cumsum(self.ap)
        return np.where(u < PartitionUnit)[0][0]


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self,ap,pm):
        #pass
        self.obj = Categorical(ap)
        self.pm = pm
   
    def sample(self):
        x = self.obj.sample()
        return self.pm[x].sample()
    
    

def main():
    number = 10000
    #Categorical Distribution
    AP = [0.1,0.1,0.3,0.3,0.2]
    cat = Categorical(AP)
    CAT = [cat.sample() for _ in range(number)]
    plt.hist(CAT)
    plt.title('Categorical Distribution')
    plt.show()
    
    #Univariate normal distribution
    mu = 1
    sigma = 1
    normU= UnivariateNormal(mu,sigma)
    normUsample = [normU.sample() for _ in range(number)]
    plotRange = np.arange(mu-3.5*sigma, mu+3.5*sigma,0.4)
    plt.hist(normUsample, plotRange)
    plt.title('Univariate normal distribution')
    plt.show()
    
    #Multivariate normal distribution
    Mu = np.array([1, 1])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    normM = MultiVariateNormal(Mu, Sigma)
    normMsample = [normM.sample() for _ in range(number)]
    X,Y = zip(*normMsample)
    plt.scatter(X,Y, alpha=0.2)
    plt.title('Multivariate normal distribution')
    plt.show()  


if __name__ == '__main__':
    main()

