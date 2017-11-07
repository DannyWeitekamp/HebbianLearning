from hebbian import HebbianLayer
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

def plot_images(images):
    from matplotlib import pyplot as plt
    # plt.axis('off')
    fig, axes = plt.subplots(15,15,sharex=True,sharey=True)
    for k,im in enumerate(images):
        i,j = k // 15, k % 15
        ax = axes[i,j]
        ax.set_axis_off()
        ax.matshow(im.data.numpy())#.reshape(7,7))
    for k in range(k,15*15):
        i,j = k // 15, k % 15
        axes[i,j].set_axis_off()
            # fig.colorbar(im, ax=ax)
    plt.show()

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
                    add_at_sum=.2,
                    add_max_corr = .15,
                    average_activation=.15,
                    conic_filters=True,
                    center=True)

# tracemalloc.start()



# for epoch in range(20):  # loop over the dataset multiple times

IMAGES_TO_SEE = 20000

images_seen = 0 
for i, data in enumerate(trainloader, 0):
    inputs, labels = data

    inputs = Variable(inputs,requires_grad=False,volatile=True)
    # print(inputs.size())
    if(not net.initialized): net.initialize({"patches":net.get_patches(inputs)})
    
    _,a = net.AHL_update(inputs+.01)
    images_seen += inputs.size()[0]
    print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net.num_neurons))
    # if(images_seen % 5000 == 0):
    #     sub = net.conv.weight[:,:49]
    #     sub.contiguous()
    #     plot_images(sub.view(*(list(sub.size()[:1]) + [7,7])))
    #     im = a[0].permute(2,0,1)
    #     im = F.max_pool2d(im,(2,2))
    #     plot_images(im)
    if(images_seen >= IMAGES_TO_SEE):
        break

# print(list(net.conv.weight.size()[:1]) + [7,7])
# sub = net.conv.weight[:,:49]
# plot_images(sub.view(*(list(sub.size()[:1]) + [7,7])))



net2 = HebbianLayer(net.num_neurons,
                    filter_size=(4,4),
                    max_active_post_neurons=1,
                    add_at_sum=.2,
                    add_max_corr = .2,
                    average_activation=.15,
                    conic_filters=True,
                    center=True)
# net2 = HebbianLayer(net.num_neurons,filter_size=(5,5),max_active_post_neurons=1)

images_seen = 0 
for i, data in enumerate(trainloader, 0):
    inputs, labels = data

    inputs = Variable(inputs,requires_grad=False,volatile=True)
    _,_,inputs = net.forward(inputs)
    inputs = F.max_pool2d(inputs,(2,2))
    print(inputs.size())
    if(not net2.initialized): net2.initialize({"patches":net2.get_patches(inputs)})
    
    net2.AHL_update(inputs)
    images_seen += inputs.size()[0]
    print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net2.num_neurons))
    if(images_seen >= IMAGES_TO_SEE):
        break



# net3 = HebbianLayer(net2.num_neurons,filter_size=(3,3),max_active_post_neurons=1)
net3 = HebbianLayer(net2.num_neurons,
                    filter_size=(2,2),
                    max_active_post_neurons=1,
                    add_at_sum=.2,
                    add_max_corr = .5,
                    average_activation=.1,
                    conic_filters=True,
                    center=True)

images_seen = 0 
for i, data in enumerate(trainloader, 0):
    inputs, labels = data

    inputs = Variable(inputs,requires_grad=False,volatile=True)
    _,_,inputs = net.forward(inputs)
    inputs = F.max_pool2d(inputs,(2,2))
    _,_,inputs = net2.forward(inputs)
    inputs = F.max_pool2d(inputs,(2,2))

    print(inputs.size())
    if(not net3.initialized): net3.initialize({"patches":net3.get_patches(inputs)})
    
    net3.AHL_update(inputs)
    images_seen += inputs.size()[0]
    print("IMAGES SEEN: %d , NEURONS: %d" % (images_seen,net3.num_neurons))
    if(images_seen >= IMAGES_TO_SEE):
        break

print("NETWORK OUT")

## training
model = MLPNet(net3.num_neurons,3,3)
# model = MLPNet(net2.num_neurons,4,4)
# model = MLPNet(net.num_neurons,11,11)

optimizer = optim.Adam(model.parameters(), lr=0.005)
for epoch in xrange(10):
    # trainning
    for batch_idx, (x, target) in enumerate(trainloader):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        _,_,x = net.forward(x)
        # print(x.size())
        x = F.max_pool2d(x,(2,2))
        # print(x.size())
        _,_,x = net2.forward(x)
        x = F.max_pool2d(x,(2,2))
        # print(x.size())
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
        x = F.max_pool2d(x,(2,2))
        _,_,x = net2.forward(x)
        x = F.max_pool2d(x,(2,2))
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
    for batch_idx, (x, target) in enumerate(trainloader):
        optimizer.zero_grad()
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