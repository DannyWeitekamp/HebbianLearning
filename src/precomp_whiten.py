from whiten import compute_zca_whitening,store_zca
from hebbian import HebbianLayer
import torchvision.transforms as transforms
from torch.autograd import Function,Variable
import torchvision
import torch
#from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np

# Basically re-maps the [0,1] pixel to the [-1,1] range so that mean is 0.
transform = transforms.Compose(
    [transforms.ToTensor()])
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Getting the data
batch_size = 1000
epsilon = 1e-1

trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True,
                                        download=True, transform=transform)
# t = trainset[:]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
# d,l = [x for x in trainloader][0]
# d = Variable(d)
# ipca = IncrementalPCA(batch_size=batch_size)
net = HebbianLayer(3,center=True,normalize=False)
print(len(trainset))
sigma = 0 

N = 0
for i,(d,l) in enumerate(trainloader):
	# if(i == 1):
	# 	break
	d = Variable(d)
	X = net.get_patches(d).permute(0,2,3,1)
	X.contiguous()
	print(X.size())
	X = X.view(-1,X.size()[-1])
	N += len(X)
	print(X.size())
	# N = X.size()[0]
	X = np.transpose(np.array(X.data.numpy(),dtype=np.double))
	# Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
	sigma = sigma + np.dot(X,X.T)#/N#np.cov(X, rowvar=True) # [M x M]
	# ipca.partial_fit(sigma)

print(sigma)
print(len(trainset))
sigma = sigma / N#len(trainset)
print(sigma)

U,S,V = np.linalg.svd(sigma)

#U = ipca.components_
#S = ipca.singular_values_

ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
print(type(ZCAMatrix))
store_zca("cifar/cifar_7x7.h5",ZCAMatrix)

print(d.size())



# print(patches.size())



# compute_zca_whitening()