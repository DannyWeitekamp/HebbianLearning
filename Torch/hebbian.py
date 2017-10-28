import torch
import torchvision
from torch.autograd import Function,Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import random,itertools
# import tracemalloc

try:
    xrange
except NameError:
    xrange = range

from torch.autograd import Variable

def normalize_patches(patches,):
    norm = torch.sqrt(torch.sum((patches**2),dim=1,keepdim=True))
    is_zero = (norm == 0.0).float()
    patches += is_zero*(1.0/np.sqrt(np.prod(norm.size()[2:])))
    norm += is_zero 
    patches = torch.div(patches,norm)
    return patches

class HebbianLayer(nn.Module):
    def __init__(self,in_channels,#, out_channels,
                            filter_size=(7,7),
                            add_max_corr=.7,
                            add_at_sum=.75,
                            max_active_post_neurons=3,
                            average_activation=.2,
                            competition_thresh=.9,
                            learning_rate=.01,
                            num_initial_filters=4):
        super(HebbianLayer, self).__init__()
        self.in_channels = in_channels
        self.add_max_corr = add_max_corr
        self.add_at_sum = add_at_sum
        self.max_active_post_neurons = max_active_post_neurons
        self.average_activation = average_activation
        self.competition_thresh = competition_thresh
        self.learning_rate = learning_rate
        self.filter_size = filter_size
        self.num_initial_filters=num_initial_filters
        self.patch_dimension = in_channels*int(np.prod(filter_size))
        # self.conv = nn.Conv2d(in_channels, out_channels, filter_size,stride=filter_size,bias=False)
        self.conv = nn.Conv2d(self.patch_dimension, self.num_initial_filters, 1,stride=1,bias=False)
        # self.conv = nn.Conv1d(in_channels*np.prod(filter_size), out_channels, 1,stride=1,bias=False)

        
        self.proj_conv = nn.Conv2d(in_channels, self.patch_dimension, filter_size,bias=False)
        self.proj_conv.weight.data = torch.eye(self.patch_dimension).view(self.patch_dimension,in_channels,*filter_size)

        # print("WEIGHT", self.proj_conv.weight.size())
        
        # print("WEIGHT", self.proj_conv.weight.size())
        # self.running_avg = Parameter(torch.Tensor(?))
        # self.b = Parameter(torch.Tensor(?)) 
        # if bias:
        self.initialized = False

    def initialize(self,weights={}):
        if(0): #TODO: Load actual weights from file
            pass
        else:
            print("INITIALIZE")
            print(self.conv.weight.size())
            if("patches" in weights):
                # Initialize from a set of patches
                patches = weights["patches"]
                patches_size = np.array(patches.size())
                nums = random.sample(xrange(0,int(np.prod(patches_size[[0,2,3]]))), self.num_initial_filters)
                a,b,c = np.unravel_index(nums,patches_size[[0,2,3]])
                inds = [()]
                conv_weights = torch.cat([patches[p:p+1,:,q:q+1,r:r+1] for p,q,r in zip(a,b,c)],dim=0)
            else:
                # Initialize from random numbers
                conv_weights = torch.rand(self.num_initial_filters,int(np.prod(self.filter_size)),1,1)
            # print(self.conv.weight.data.shape)
            conv_weights = normalize_patches(conv_weights)
            self.conv.weight.data = conv_weights.data
            # print("BEEP",self.conv.weight.data.shape)
            self.bias = Variable(torch.Tensor(self.num_initial_filters),requires_grad=False,volatile=True)
            self.running_avg = Variable((self.average_activation)*torch.ones(self.num_initial_filters),requires_grad=False,volatile=True)

        # raise RuntimeError()
        self.initialized = True
        
    
    def get_patches(self,x):
        patches = self.proj_conv(x)


        # print("WAWA",patches.size())
        patch_mean = torch.mean(patches, dim=1, keepdim=True)
        # print(patch_mean)
        patches -= patch_mean

        patches = normalize_patches(patches)
        # norm = torch.sum((patches**2),dim=1,keepdim=True)
        # print("MAXI", norm.data[0][0])
        # print("THIS",patches.size())
        
        # (255*patches[0,0].data).int()
        # print("AAA")

        if(0):
            fig, axes = plt.subplots(10,10,sharex=True,sharey=True)
            print("BB")
            for i in range(10):
                for j in range(10):
                    print(i,j)
                    ax = axes[i,j]
                    im = ax.matshow(patches[0,:,i+11,j+11].data.numpy().reshape(7,7))
                    fig.colorbar(im, ax=ax)
                    # fig[i,j].colorbar()
            # plt.colorbar()
            plt.show()
            raise RuntimeError()
        # s = patches.size()
        # print(s)
        return patches

    def forward(self, x, is_patches=False):
        if(not self.initialized): self.initialize()
        if(is_patches):
            patches = x
        else:
            patches = self.get_patches(x)


        # print(patches)

        # if(type(conv) == type(None)): conv = self.conv
        # if(type(bias) == type(None)): bias = self.bias
        
        # print("MOMO", patches.size())
        # print("MOMO", Wx_.size())
        s = patches.size()
        patches.view(s[0],s[1],int(np.prod(s[2:])))
        
        # print(patches.size())
        Wx_ = self.conv(patches)
        # print("WA",Wx_.size())
        # s = Wx_.size()
        # activation_size = np.array(x.size()[2:]) - np.array(self.filter_size) + 1
        # print(activation_size)
        # Wx_ = Wx_.view(s[0],s[1],activation_size[0],activation_size[1])
        # print("MOMO", Wx_.size())
        #TODO: Standardize
        #TODO: Whiten
        # norm_Wx_  = F.normalize(Wx_,p=2,dim=1)
        a = F.relu(Wx_ - self.bias.view(1,-1,1,1))
        # print("MAX A", torch.max(a))
        # a = norm_Wx_
        p = F.avg_pool2d(a,(2,2)) 
        return patches,Wx_, a, p

    def _new_neuron_cadidates(self,Wx_,a):
        assert len(Wx_.size()) == 4
        assert len(a.size()) == 4
        
            # a.size())
        max_Wx,_ = torch.max(Wx_,dim=1)
        # print(max_Wx[0])#.size())
        sum_a = torch.sum(a,dim=1)
        # print(self.conv.weight[-1].view(7,7))

        # print(max_Wx)
        print("MIN_a: %.2f < %.2f; MIN_W: %.2f < %.2f" % (torch.min(sum_a).data[0],self.add_at_sum,torch.min(max_Wx).data[0], self.add_max_corr))
        # print("MIN_W",,)
        A = (max_Wx < self.add_max_corr)
        B = (sum_a < self.add_at_sum)
        return ((A + B) == 2).data.nonzero()
    def add_neuron(self,patches,Wx_, a):
        if(not self.initialized): self.initialize()
        filter_w, filter_h = self.filter_size
        indices = self._new_neuron_cadidates(Wx_,a)
        
        if(len(indices) > 0):
            print("Number of Candidates", indices.size()[0])
            while(len(indices) > 0):

                n,r,c = indices[np.random.randint(0,len(indices))]
                # x_patch = x[n:n+1,:,r:r+filter_h,c:c+filter_w].data
                x_patch = patches[n:n+1,:,r:r+1,c:c+1].data
                # x_patch = x_patch.contiguous()
                # print(x_patch.view(7,7))
                # print(x_patch.size())
                # x_patch = x_patch.contiguous()
                # print("SS",x_patch.size(), self.conv.weight.size())
                # x_patch_reshape = x_patch.view(1,np.prod(self.filter_size),1,1)
                # print(self.conv.weight.size(), x_patch_reshape.size())
                self.conv.weight.data = torch.cat((self.conv.weight.data,x_patch),dim=0)
                # print("S_NW", self.conv.weight.size())
                self.bias.data = torch.cat((self.bias.data,torch.zeros(1)))
                self.running_avg = torch.cat((self.running_avg,torch.ones(1)*self.average_activation))
                # print(self.running_avg.shape, )

                _,Wx_, a, _ = self.forward(patches,is_patches=True)
                
                indices = self._new_neuron_cadidates(Wx_, a)
                # print("SIZE:",indices.size(), self.conv.weight.size())
        # else:
            # print("NOPE!", self.conv.weight.size())
        return Wx_, a

    def prune_neuron(self,a):
        pass

    def delta_W(self,patches,a):
        if(not self.initialized): self.initialize()
        # print(a.size())
        W = self.conv.weight
        U,U_inds = torch.topk(a, self.max_active_post_neurons,dim=1)
        # print("U",U.size(),U_inds.size())
        # print("{ATches", patches.size())
        # print(U[:,1,1])

        # print("X", x.size())
        mask_pos = patches > 0
        mask_neg = patches < 0

        # print("W",W.size())
        filter_w, filter_h = self.filter_size
        # print(filter_w,filter_h) 
        delta = Variable(torch.zeros(W.size()),requires_grad=False,volatile=True)
        act_width,act_height = U_inds.size()[2:]
        # print(type(W),type(delta))
        for r in range(act_height):
            for c in range(act_width):
                # p_mask = mask_pos[:,:,r:r+filter_h,c:c+filter_w].unsqueeze(1)
                # n_mask = mask_neg[:,:,r:r+filter_h,c:c+filter_w].unsqueeze(1)
                # print("PM:",p_mask.size())
                # x_patch  = x[:,:,r:r+filter_h,c:c+filter_w]
                p_mask = mask_pos[:,:,r:r+1,c:c+1].unsqueeze(1)
                n_mask = mask_neg[:,:,r:r+1,c:c+1].unsqueeze(1)
                # print("PM:",p_mask.size())
                x_patch  = patches[:,:,r:r+1,c:c+1]


                top_inds = U_inds[:,:,r,c]
                # print(top_inds.size())
                # print("W",W.size())
                w_top_k = torch.cat([torch.index_select(W, 0, ti).unsqueeze(0) for ti in top_inds],0)
                # print(w_top_k.size())
                w_mask_p = w_top_k >= self.competition_thresh * w_top_k[:,0].unsqueeze(1)
                w_mask_n = w_top_k <= self.competition_thresh * w_top_k[:,-1].unsqueeze(1) 
                # print(w_mask_p.size(),p_mask.size())
                mask = (p_mask * w_mask_p + n_mask*w_mask_n) > 0

                # print(mask.size(),x_patch.size())
                temp = mask.float()*x_patch.unsqueeze(1)
                # print(temp.size(),delta.size())
                for t,inds in zip(temp,top_inds):
                    delta.index_add_(0, inds, t)
                
                # print(temp)
                # print(r-r+filter_w,c-c+filter_h)
                # print(delta[:,r:r+filter_w,c:c+filter_h].size())
                # d_sub = torch.index_select(delta,0, top_inds)
                # delta.index_add_(0, top_inds, temp)
                # print(d_sub.size(),temp.size())
                # d_sub += temp
                # print(temp)
                # print(temp)
                # print(.size())

                # print(w_top_k.size())
                # w_top_k = 
                # print(top_k,self.competition_thresh * top_k[0],self.competition_thresh * top_k[-1])
                # w_mask_p = top_k >= self.competition_thresh * top_k[0]
                # w_mask_n = top_k <= self.competition_thresh * top_k[-1]
                # print(p_mask.size(),n_mask.size(),w_mask_p.size())

                # print(top_k, max_w,min_w)

                # top_patch =   
        # print(delta)
        # print("DELTA", torch.sum(delta).data[0]/int(np.prod(delta.size())))
        self.conv.weight.data += .001* delta.data
        # print("W",W.size())
        last = W.size()[1:]
        # print(type(last))
        as_lin_trans = self.conv.weight.view(-1,int(np.prod(last)))
        self.conv.weight.data = F.normalize(as_lin_trans,dim=1).view(-1,*last).data
        # print(torch.sum(self.conv.weight.data**2))
        # print(self.bias)

        #Find the leading/tailing significant weights
        # max_weights = torch.max(W,dim=0)
        # min_weights = torch.min(W,dim=0)

        # ct = self.competition_thresh
        # print(x.size())
        # print(W.size())
        # cond1 = (x > 0.0)*(W >= ct*max_weights)
        # cond2 = (x < 0.0)*(W <= ct*min_weights)
        # mask = (cond1 + cond2) > 0
        return delta



    def update_bias(self,a):
        if(not self.initialized): self.initialize()
        # print(self.running_avg.data.shape,a.data.shape)
        # print(torch.sign(a))
        # print(torch.mean(torch.mean(torch.mean(torch.sign(a),dim=0),dim=1),dim=1))
        current = torch.mean(torch.mean(torch.mean(torch.sign(a),dim=0),dim=1),dim=1)
        # print(np.array(current.data))
        self.running_avg = 0.95*self.running_avg + .05*current
        # print(self.bias.data.shape, self.running_avg.data.shape)
        # print(.01*(self.running_avg - self.average_activation))
        # print(np.array(self.bias.data))
        self.bias += .05*(self.running_avg - self.average_activation)


        # (x > 0)*(W[U] > )

        










''' COMMENTED OUT

# class HebbianApply(Function):
#     @staticmethod
#      def forward(ctx, input, conv, bias=None):
#         Wx_ = conv(input)
#         #TODO: Standardize
#         #TODO: Whiten
#         norm_Wx_ = F.normalize(Wx_,p=2,dim=1)
#         diff = norm_Wx_ - bias if bias != None else norm_Wx_
#         a = F.relu(diff)
#         ctx.save_for_backward(Wx_,a)
#         p = F.avg_pool2d(a,(2,2)) 
#         return p

#     @staticmethod
#     def backward(ctx, grad_output):


import math
from torch.optim.optimizer import Optimizer


class HebbianOptimizer(Optimizer):
    def __init__(self, params):
        defaults = {};
        super(HebbianOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):

        print(self.param_groups)
        for group in self.param_groups:
            # print()
            print("GROUP:",group['params'])
            for p in group['params']:

                # if p.grad is None:
                    # continue
                # grad = p.grad.data
                # state = self.state[p]
                print("STATE",state)

                # State initialization
                # if len(state) == 0:
                #     state['step'] = 0
                #     state['eta'] = group['lr']
                #     state['mu'] = 1
                #     state['ax'] = grad.new().resize_as_(grad).zero_()

                # state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # # decay term
                # p.data.mul_(1 - group['lambd'] * state['eta'])

                # # update parameter
                # p.data.add_(-state['eta'], grad)

                # # averaging
                # if state['mu'] != 1:
                #     state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                # else:
                #     state['ax'].copy_(p.data)

                # # update eta and mu
                # state['eta'] = (group['lr'] /
                #                 math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                # state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return 0.0


'''