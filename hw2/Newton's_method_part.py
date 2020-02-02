import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}

#Defining auxiliary variables
X = np.array([[0,3],[1,3],[0,1],[1,1]])
y = np.array([1,1,0,0])
theta = np.array([0,  -2, 1])
ones = np.ones((X.shape[0],1), dtype = int)
XX = np.hstack((ones, X))


def sigmoid(x): 
    return 1./(1.+np.exp(-x))

def logistic_model(x,theta):
    '''Defining the logistic model'''
    scores = x.dot(theta)
    return sigmoid(scores)

def theta_lambda(theta):
    '''It would be handy to return the theta 
       for the regularization part
       with the first term setting 
       to zero'''
    thetaL = np.zeros((theta.shape))
    thetaL[1:]= theta[1:]
    return thetaL

def crossEntropy(x,y,theta, L = 0.07):
    '''Defining the cross entropy function'''
    thetaL = theta_lambda(theta)
    h_theta = logistic_model(x, theta)
    delta1, delta2 = np.log(h_theta), np.log(1-h_theta)
    return 1/len(y)*(-(delta1.dot(y)+
           delta2.dot(1-y))+ L*0.5* thetaL.dot(thetaL))    

def crossEntropyGradient(x,y,theta, L = 0.07):
    thetaL = theta_lambda(theta)
    h_theta = logistic_model(x, theta)
    error = h_theta -y
    return (np.matmul(XX.T, error)+L*thetaL)/len(y)

def crossEntropyHessian(x,y,theta, L = 0.07):
    m, n = x.shape
    h_theta = logistic_model(x, theta)
    I =np.identity(n)
    S  = np.diag(h_theta *(1-h_theta))
    return (x.T.dot(S.dot(x))+ L* I)/m
    
gradient = lambda theta: crossEntropyGradient(XX, y, theta)
hessian = lambda theta: crossEntropyHessian(XX, y, theta)

def newton_descent(init, step_sizes, grad, hessian):
    Theta = [init]
    for step in step_sizes:
        Hinv = np.linalg.pinv(hessian(Theta[-1]))
        Theta.append(Theta[-1] - step * Hinv.dot(grad(Theta[-1])))
    return Theta

def gradient_descent(init, step_sizes, grad):    
    Theta = [init]
    for step in step_sizes:
        Theta.append(Theta[-1] - step * grad(Theta[-1]))
    return Theta


theta_newton = newton_descent(theta, np.ones(5), 
               gradient, hessian)
theta_gd = gradient_descent(theta, np.ones(5), gradient)

def error_plot(ys1, ys2, yscale='log'):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys1)), ys1, **kwargs, label='GD')
    plt.plot(range(len(ys2)), ys2, **kwargs, label='Newton')
    plt.legend()
    plt.show()

def main(n):   
    step_sizes = np.ones(n)
    theta_newton = newton_descent(theta, step_sizes, 
                                 gradient, hessian)
    theta_gd = gradient_descent(theta, step_sizes, gradient)
    print("Newton Method's theta")
    print(theta_newton)
    print("Gradient Descent's theta")
    print(theta_gd)
    error_plot([crossEntropy(XX,y,theta, L = 0.07) 
                for theta in theta_gd], 
                [crossEntropy(XX,y,theta, L = 0.07) 
                for theta in theta_newton])

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    main(n)


