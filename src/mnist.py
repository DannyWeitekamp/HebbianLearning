from hebbian import HebbianLayer
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
# import tracemalloc
import objgraph
from sklearn.svm import LinearSVC,SVC
# (Pdb) objgraph.show_most_common_types(limit=20)

## network
class MLPNet(nn.Module):
    def __init__(self,neurons,input_w,input_h):
        super(MLPNet, self).__init__()
        self.neurons = neurons
        self.input_w = input_w
        self.input_h = input_h
        self.fc1 = nn.Linear(self.input_w*self.input_h*self.neurons, 10)
        # self.fc2 = nn.Linear(500, 256)
        # self.fc3 = nn.Linear(256, 10)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x, target):
        x = x.view(-1, self.input_w*self.input_h*self.neurons)
        x = F.softmax(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        loss = self.ceriation(x, target)
        return x, loss
    def name(self):
        return 'mlpnet'


def unnormalize(img):
    img = img / 2 + 0.5
    return img

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_images(images,show=True):
    from matplotlib import pyplot as plt
    # plt.axis('off')
    fig, axes = plt.subplots(15,15,sharex=True,sharey=True)
    for k,im in enumerate(images):
        i,j = k // 15, k % 15
        ax = axes[i,j]
        ax.set_axis_off()
        data = im if not isinstance(im,Variable) else im.data
        data = data if not isinstance(data,torch.Tensor) else data.squeeze().numpy()
        print(data.shape)
        ax.matshow(data)#.reshape(7,7))
    for k in range(k,15*15):
        i,j = k // 15, k % 15
        axes[i,j].set_axis_off()
            # fig.colorbar(im, ax=ax)
    if(show):plt.show()

# Basically re-maps the [0,1] pixel to the [-1,1] range so that mean is 0.
transform = transforms.Compose(
    [transforms.ToTensor(),])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Getting the data
batch_size = 100

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

net = HebbianLayer(1,max_active_post_neurons=1,
                    add_at_sum=.6,
                    add_max_corr = .7,
                    average_activation=.2,
                    conic_filters=False,
                    prune_max_corr=.6,
                    spatial_locality=0.0,
                    center=True)

# tracemalloc.start()



# for epoch in range(20):  # loop over the dataset multiple times

IMAGES_TO_SEE = 60000
pool = lambda x: F.max_pool2d(x,(2,2))
run2,run3 = True,True

images_seen = 0 
for i, data in enumerate(trainloader, 0):
    inputs, labels = data

    inputs = Variable(inputs,requires_grad=False,volatile=True)
    # print(inputs.size())
    if(not net.initialized): net.initialize({"patches":net.get_patches(inputs)})
    
    _,a = net.AHL_update(inputs)
    images_seen += inputs.size()[0]
    print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net.num_neurons))
    # if(images_seen == IMAGES_TO_SEE):
    #     print(net.correlation_matrix)
    #     sub = net.conv.weight[:,:49].view(-1,7,7)
    #     sub.contiguous()
    #     # print(torch.max(a))
    #     plot_images(sub,show=False)
    #     im = a[0].permute(2,0,1)
    #     # im = pool(im)
    #     plot_images(im)
    if(images_seen >= IMAGES_TO_SEE):
        break


layer1_X,layer1_y = [],[]
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = Variable(inputs,requires_grad=False,volatile=True)
    # a = inputs
    patches,Wx_, a = net.forward(inputs)
    a = pool(a)
    layer1_X.append(np.array(a.data.numpy(),dtype=np.float64))
    layer1_y.append(np.array(labels.numpy(),dtype=np.float64))
    if(i%100 == 0):
        print(i*batch_size)
    if(i >= 299):
       break
layer1_X,layer1_y = np.concatenate(layer1_X),np.concatenate(layer1_y)
layer1_X = layer1_X.reshape(layer1_X.shape[0],-1)

print("FIT ON",layer1_X.shape,layer1_y.shape,layer1_X.flags,layer1_X.dtype)
# raise RuntimeError()
svm = SVC(C=100,kernel='linear')
svm.fit(layer1_X,layer1_y)
layer1_X,layer1_y = None,None

layer1_X_val,layer1_y_val = [],[]
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = Variable(inputs,requires_grad=False,volatile=True)
    # a = inputs
    patches,Wx_, a = net.forward(inputs)
    a = pool(a)
    layer1_X_val.append(np.array(a.data.numpy(),dtype=np.float64))
    layer1_y_val.append(np.array(labels.numpy(),dtype=np.float64))
    if(i%100 == 0):
        print(i*batch_size)
    # if(images_seen >= IMAGES_TO_SEE):
    #     break
layer1_X_val,layer1_y_val = np.concatenate(layer1_X_val),np.concatenate(layer1_y_val)
layer1_X_val = layer1_X_val.reshape(layer1_X_val.shape[0],-1)

# print(layer1_X.shape,layer1_y.shape)

print("TEST ON",layer1_X_val.shape,layer1_y_val.shape)
print("TEST ACC", svm.score(layer1_X_val,layer1_y_val))


raise RuntimeError()
# print(list(net.conv.weight.size()[:1]) + [7,7])
# sub = net.conv.weight[:,:49]
# plot_images(sub.view(*(list(sub.size()[:1]) + [7,7])))


if(run2):
    net2 = HebbianLayer(net.num_neurons,
                        filter_size=(4,4),
                        max_active_post_neurons=1,
                        add_at_sum=1.0,
                        add_max_corr = .7,
                        average_activation=.2,
                        conic_filters=False,
                        center=True)
    # net2 = HebbianLayer(net.num_neurons,filter_size=(5,5),max_active_post_neurons=1)

    images_seen = 0 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = Variable(inputs,requires_grad=False,volatile=True)
        _,_,inputs = net.forward(inputs)
        inputs = pool(inputs)
        print(inputs.size())
        if(not net2.initialized): net2.initialize({"patches":net2.get_patches(inputs)})
        
        net2.AHL_update(inputs)
        images_seen += inputs.size()[0]
        print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net2.num_neurons))
        if(images_seen >= IMAGES_TO_SEE):
            break


if(run3):
    # net3 = HebbianLayer(net2.num_neurons,filter_size=(3,3),max_active_post_neurons=1)
    net3 = HebbianLayer(net2.num_neurons,
                        filter_size=(2,2),
                        max_active_post_neurons=1,
                        add_at_sum=2.8,
                        add_max_corr = .6,
                        average_activation=.2,
                        conic_filters=False,
                        center=True)

    images_seen = 0 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs = Variable(inputs,requires_grad=False,volatile=True)
        _,_,inputs = net.forward(inputs)
        inputs = pool(inputs)
        _,_,inputs = net2.forward(inputs)
        inputs = pool(inputs)

        print(inputs.size())
        if(not net3.initialized): net3.initialize({"patches":net3.get_patches(inputs)})
        
        net3.AHL_update(inputs)
        images_seen += inputs.size()[0]
        print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net3.num_neurons))
        if(images_seen >= IMAGES_TO_SEE):
            break

print("NETWORK OUT")

last_net = net3 if run3 else (net2 if run2 else net)
print(last_net.num_neurons)
## training

if(run3):
    model = MLPNet(last_net.num_neurons,3,3)
elif(run2):
    model = MLPNet(last_net.num_neurons,4,4)
else:
    model = MLPNet(last_net.num_neurons,11,11)  
# 
# model = MLPNet(last_net.num_neurons,4,4)


samples = [trainset[i] for i in range(200)]
samples_per_class = 10


ss = []
for i in range(10):
    to_add = [(x.unsqueeze(0),torch.LongTensor([l])) for x,l in samples if l==i][:samples_per_class]
    assert(len(to_add) == samples_per_class, "got %d requested %d"% (len(to_add),samples_per_class))
    ss += to_add

# print(ss)
x_sub,target_sub = torch.cat([x for x,y in ss]),torch.cat([y for x,y in ss])

# plot_images(x_sub)
# print(x_sub.size(),target_sub.size())
tot = x_sub.size()[0]
def sub_sample_loader():
    repeats = max(10000/batch_size,1)
    print("REPEATS",repeats)
    for i in range(repeats):
        for j in range(0,tot,batch_size):
            # print()
            end = max(min(j+batch_size,tot-j),0)
            # print(j,end,tot)
            if(j == end):break
            # print(x.size())
            yield x_sub[j:end], target_sub[j:end]

class RepSubSampleLoader():
    def __iter__(self):
        return sub_sample_loader()


loader = RepSubSampleLoader()
print(loader)
# loader = trainloader

# print(x,target)
# x,target = np.array(x,dtype=np.float32),np.array(target,dtype=np.int)
# x,target = torch.Tensor(x),torch.Tensor(target)
# X, target = Variable(x), Variable(target)

optimizer = optim.Adam(model.parameters(), lr=0.005)
for epoch in xrange(10):
    # trainning
    print(epoch)
    for batch_idx, (x,target) in enumerate(loader):#range(500):
        # print(target)
        # print(type(x),type(target))
        optimizer.zero_grad()
        # print(x,target)
        if(not isinstance(x, Variable)):
            X, target = Variable(x), Variable(target)
        _,_,x = net.forward(X)
        # print(x.size())
        x = pool(x)
        # print(x.size())
        if(run2):
            _,_,x = net2.forward(x)
            x = pool(x)
                # print(x.size())
        if(run3):
            _,_,x = net3.forward(x)
        # print(x.size())
        _, loss = model(Variable(x.data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])
        # break;
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(testloader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        _,_,x = net.forward(x)
        x = pool(x)
        if(run2):
            _,_,x = net2.forward(x)
            x = pool(x)
        if(run3):
            _,_,x = net3.forward(x)

        score, loss = model(Variable(x.data), target)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(testloader)/batch_size
    ave_loss /= len(testloader)
    print '==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy)


print("DIRECT SUPERVISED")

## training
model = MLPNet(1,28,28)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in xrange(10):
    # trainning
    # samples = [next(trainloader) for x in range(5)]
    for batch_idx, (x, target) in enumerate(trainloader):
        optimizer.zero_grad()
        if(not isinstance(x, Variable)):
            # print(type(x),Variable)
            x, target = Variable(x), Variable(target)
        # _,_,x = net.forward(x)
        # x = F.max_pool2d(x,(2,2))
        # _,_,x = net2.forward(x)
        # x = F.max_pool2d(x,(2,2))
        # _,_,x = net3.forward(x)

        _, loss = model(Variable(x.data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])
        # break;
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(testloader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        # _,_,x = net.forward(x)
        # x = F.max_pool2d(x,(2,2))
        # _,_,x = net2.forward(x)
        # x = F.max_pool2d(x,(2,2))
        # _,_,x = net3.forward(x)

        score, loss = model(Variable(x.data), target)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(testloader)/batch_size
    ave_loss /= len(testloader)
    print '==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy)

        # objgraph.show_most_common_types(limit=20)
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

print('Finished Training')

# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# print('Accuracy of the network on the %d test images: %f %%' % (
#     total, 100.0 * correct / total))