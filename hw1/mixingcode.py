import matplotlib.pyplot as plt
import sampler
import numpy as np


I = np.array([[1,0], [0,1]])

N1 = sampler.MultiVariateNormal(np.array([1,1]), I)
N2 = sampler.MultiVariateNormal(np.array([-1,1]), I)
N3 = sampler.MultiVariateNormal(np.array([1,-1]), I)
N4 = sampler.MultiVariateNormal(np.array([-1,-1]), I)
weights = [0.25, 0.25, 0.25, 0.25]

obj = sampler.MixtureModel(weights, [N1, N2, N3, N4])

Allx = []
Ally = []

xInside=[]
yInside=[]

Counter = 0
for _ in range(10000):
    x, y = (obj.sample())
    Allx.append(x)
    Ally.append(y)
    if np.sqrt((x-0.1)**2 + (y-0.2)**2) <= 1:
        Counter +=1
        xInside.append(x)
        yInside.append(y)

total = len(Allx)
prob = float(Counter)/float(total) #Freq. 

print(prob)


plt.plot(Allx, Ally, 'bo', alpha= 0.1)
plt.plot(xInside, yInside, 'r,',  alpha=1)
plt.title('Mixture sampling exercise, Prob=%f' %prob)

plt.savefig('sample.png')

