from hebbian import HebbianLayer
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

def unnormalize(img):
    img = img / 2 + 0.5
    return img

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Basically re-maps the [0,1] pixel to the [-1,1] range so that mean is 0.
transform = transforms.Compose(
    [transforms.ToTensor(),])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Getting the data

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)

net = HebbianLayer(1)

# tracemalloc.start()


for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):


        # get the inputs
        inputs, labels = data
        # print inputs.numpy().shape

        # wrap them in Variable
        inputs = Variable(inputs,requires_grad=False,volatile=True)
        if(not net.initialized): net.initialize({"patches":net.get_patches(inputs)})
        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward + backward + optimize
        
        print(i)
        patches,Wx_, a, p = net(inputs)
        Wx_, a = net.add_neuron(patches,Wx_,a)
        net.delta_W(patches,a)
        net.update_bias(a)
        net.prune(a)

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)
        
        # for x,ai in zip(inputs,a):
            
            # print(outputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # print statistics
        # # running_loss += loss.data[0]
        # if i % 100 == 99:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 100))
        #     running_loss = 0.0

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