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

def normalize_patches(patches):
    norm = torch.sqrt(torch.sum((patches**2),dim=1,keepdim=True))
    is_zero = (norm == 0.0).float()
    patches += is_zero*(1.0/np.sqrt(np.prod(norm.size()[2:])))
    norm += is_zero 
    patches = torch.div(patches,norm)
    return patches

def plot_patches(patches):
    fig, axes = plt.subplots(10,10,sharex=True,sharey=True)
    for i in range(10):
        for j in range(10):
            ax = axes[i,j]
            im = ax.matshow(patches[0,:,i+11,j+11].data.numpy().reshape(7,7))
            fig.colorbar(im, ax=ax)
    plt.show()

class HebbianLayer(nn.Module):
    def __init__(self,in_channels,
                            filter_size=(7,7),
                            add_max_corr=.7,
                            add_at_sum=.75,
                            max_active_post_neurons=3,
                            average_activation=.2,
                            competition_thresh=.9,
                            learning_rate=.01,
                            prune_max_corr = .8,
                            num_initial_filters=4,
                            prune_subsamples=2500):
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
        self.prune_max_corr = prune_max_corr
        self.patch_dimension = in_channels*int(np.prod(filter_size))
        self.prune_subsamples = prune_subsamples

        self.conv = nn.Conv2d(self.patch_dimension, self.num_initial_filters, 1,stride=1,bias=False)

        self.proj_conv = nn.Conv2d(in_channels, self.patch_dimension, filter_size,bias=False)
        self.proj_conv.weight.data = torch.eye(self.patch_dimension).view(self.patch_dimension,in_channels,*filter_size)

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
            conv_weights = normalize_patches(conv_weights)
            self.conv.weight.data = conv_weights.data
            self.bias = Variable(torch.Tensor(self.num_initial_filters),requires_grad=False,volatile=True)
            self.running_avg = Variable((self.average_activation)*torch.ones(self.num_initial_filters),requires_grad=False,volatile=True)


        #Initialize Covariance Matrix - to roughly zero matrix, assume small variance to prevents division by zero.
        self.covariance_matrix = Variable(1e-2*torch.eye(self.num_initial_filters,self.num_initial_filters),requires_grad=False,volatile=True)
        # raise RuntimeError()
        self.initialized = True
        self.num_neurons = self.num_initial_filters
        
    
    def get_patches(self,x):
        patches = self.proj_conv(x)

        #CENTER
        patch_mean = torch.mean(patches, dim=1, keepdim=True)
        patches -= patch_mean

        #TODO: WHITEN

        #L2 Normalize
        patches = normalize_patches(patches)
            
        return patches
    def _assert_NHWC(self,a,Wx_=None):
        assert a.size()[-1] == self.num_neurons, "calling function requires NHWC format got size: %s" % str(a.size())
        if(type(Wx_) != type(None)): assert Wx_.size()[-1] == self.num_neurons,"calling function requires NHWC format got size: %s" % str(a.size())
    def _NCHW_to_NHWC(self,x):
        x = x.permute(0,2,3,1)
        x.contiguous()
        return x

    def _NHWC_to_NCHW(self,x):
        x = x.permute(0,2,3,1)
        x.contiguous()
        return x

    def AHL_update(self,inputs):
        patches,Wx_, a = self.forward(inputs,dim_order='NHWC')
        Wx_, a = self.add_neuron(patches,Wx_,a)
        self.update_weights(patches,a)
        self.update_bias(a)
        self.prune(a)
        return Wx_, a 

    def forward(self, x, is_patches=False, dim_order='NCHW'):
        if(not self.initialized): self.initialize()
        if(is_patches):
            patches = x
        else:
            patches = self.get_patches(x)

        #Get the linear convolutional activations
        Wx_ = self.conv(patches)

        #Make sure that we evalate in the correct mode for efficient learning
        if(dim_order == 'NHWC'):
            Wx_ = self._NCHW_to_NHWC(Wx_)
            expanded_bias = self.bias.view(1,1,1,-1)
        elif(dim_order == "NCHW"):
            expanded_bias = self.bias.view(1,-1,1,1)
        else:
            raise RuntimeError("dim_order not recognized %s only accepts 'NCHW' or 'NCHW'" % dim_order)

        a = F.relu(Wx_ - expanded_bias)

        return patches,Wx_, a

    def _new_neuron_cadidates(self,Wx_,a):
        assert len(Wx_.size()) == 4
        assert len(a.size()) == 4
        
        max_Wx,_ = torch.max(Wx_,dim=-1)
        sum_a = torch.sum(a,dim=-1)
        
        print("MIN_a: %.2f < %.2f; MIN_W: %.2f < %.2f" % (torch.min(sum_a).data[0],self.add_at_sum,torch.min(max_Wx).data[0], self.add_max_corr))
        
        A = (max_Wx < self.add_max_corr)
        B = (sum_a < self.add_at_sum)
        return ((A + B) == 2).data.nonzero()

    def _add_neuron(self,x_patch):
        self.conv.weight.data = torch.cat((self.conv.weight.data,x_patch),dim=0)
        self.bias.data = torch.cat((self.bias.data,torch.zeros(1)))
        self.running_avg = torch.cat((self.running_avg,torch.ones(1)*self.average_activation))
        cov_shape = self.covariance_matrix.size()
        self.covariance_matrix = F.pad(self.covariance_matrix.view(1,1,*cov_shape), (0,1,0,1), 'constant', 0)
        self.covariance_matrix = self.covariance_matrix.view(cov_shape[0]+1,cov_shape[1]+1)
        self.covariance_matrix[-1,-1] = 1e-2
        self.num_neurons += 1
                
    def add_neuron(self,patches,Wx_, a):
        if(not self.initialized): self.initialize()
        self._assert_NHWC(a,Wx_)
        filter_w, filter_h = self.filter_size
        indices = self._new_neuron_cadidates(Wx_,a)
        
        if(len(indices) > 0):
            # print("Number of Candidates", indices.size()[0])
            while(len(indices) > 0):

                n,r,c = indices[np.random.randint(0,len(indices))]
                x_patch = patches[n:n+1,:,r:r+1,c:c+1].data
                
                self._add_neuron(x_patch)
                print("Nueron Added: %d" % self.num_neurons)


                _,Wx_, a = self.forward(patches,is_patches=True,dim_order="NHWC")
                
                indices = self._new_neuron_cadidates(Wx_, a)

        return Wx_, a

    def _remove_neurons(self,to_remove):
        if(len(to_remove) > 0):
            assert max(to_remove) < self.num_neurons, "index %r out of bounds" % max(to_remove)
            assert min(to_remove) >= 0, "index %r out of bounds" % min(to_remove)

            to_keep = torch.LongTensor([i for i in range(self.num_neurons) if not i in to_remove])
            self.covariance_matrix = self.covariance_matrix[:,to_keep][to_keep,:] 
            self.running_avg = self.running_avg[to_keep]
            self.bias = self.bias[to_keep]
            self.conv.weight.data = self.conv.weight.data[to_keep]
            self.num_neurons -= len(to_remove)

    def prune(self,a):
        self._assert_NHWC(a)
        a = a.view(-1,self.num_neurons)
        
        new_cov = 0
        L = a.size()[0]

        if(self.prune_subsamples > 0):
            a = a[torch.LongTensor(np.random.choice(xrange(L), self.prune_subsamples, replace=False))]
            L = a.size()[0]

        subbatch_size = 100000/(self.num_neurons**2)
        for i in range(0,L,subbatch_size): 
            end = min(L,i+subbatch_size)
            
            a_batch = a[i:end]
            temp = torch.bmm(a_batch.unsqueeze(2),a_batch.unsqueeze(1))
            temp = torch.sum(temp,0)
            new_cov = new_cov + temp
        new_cov /= L
        self.covariance_matrix = .95*self.covariance_matrix + .05*new_cov
        
        std = torch.sqrt(torch.diag(self.covariance_matrix))
        den = std.unsqueeze(0)*std.unsqueeze(1)
        correlation_matrix = self.covariance_matrix / den
        
        correlation_matrix = torch.triu(correlation_matrix,diagonal=1)

        to_remove = []
        while(True):
            mx,arg_mx1 = torch.max(correlation_matrix,dim=0)
            mx,arg_mx2 = torch.max(mx,dim=0)
            mx, arg_mx1,arg_mx2 = mx.data[0],arg_mx1[arg_mx2].data[0],arg_mx2.data[0]
            print(mx,arg_mx1,arg_mx2)
            assert mx == correlation_matrix[arg_mx1,arg_mx2].data[0]
            if(mx <= self.prune_max_corr):
                break
            r = max(arg_mx1,arg_mx2)
            correlation_matrix[:,r] = correlation_matrix[:,r] - 1.0
            correlation_matrix[r,:] = correlation_matrix[:,r] - 1.0
            to_remove.append(r)

        self._remove_neurons(to_remove)
        

    def update_weights(self,patches,a):
        self._assert_NHWC(a)
        if(not self.initialized): self.initialize()
        W = self.conv.weight


        if(1):
            delta = Variable(torch.zeros(W.size()[:2]),requires_grad=False,volatile=True)
            #Get the indicies of the top filters activated for each patch
            print("A",a.size())

            # #Put a, and patches in (total patches,number neurons) format so
            # # we can apply the update in sub-batches to minimize cache hits
            # a = a.permute(0,2,3,1) #size: (nn,N,ph,pw)
            # a.contiguous()
            a = a.view(-1,self.num_neurons) #size (total patches,nn)

            patches = patches.permute(0,2,3,1)
            patches.contiguous()
            patches = patches.view(-1,patches.size()[-1]) #size (total patches,nn)
            # print(a.size(),patches.size())

            L = patches.size()[0]
            subbatch_size = 100000/self.num_neurons
            for i in range(0,L,subbatch_size): 
                end = min(L,i+subbatch_size)
                s = end-i
                a_batch,patches_batch = a[i:end],patches[i:end].unsqueeze(1) #size:(subbatch_size*Kw,)
                U,U_inds = torch.topk(a_batch, self.max_active_post_neurons,dim=1) #size:(subbatch_size*Kw,)
                U_inds_flat = U_inds.view(-1) #size:(subbatch_size*Kw,)
                
                u_size = U_inds.size() 
                W_topk = W[U_inds_flat].view(s,self.max_active_post_neurons,-1) #size:(subbatch_size,Kw,fh*fw)
                W_min,_ = torch.min(W_topk,dim=1,keepdim=True) #size:(subbatch_size,1,fh*fw)
                W_max,_ = torch.max(W_topk,dim=1,keepdim=True) #size:(subbatch_size,1,fh*fw)
                                
                p_mask = (patches_batch > 0) #size:(subbatch_size,1,fh*fw)
                n_mask = (patches_batch < 0) #size:(subbatch_size,1,fh*fw)
                w_mask_p = W_topk >= self.competition_thresh * W_max #size:(subbatch_size,Kw,fh*fw)
                w_mask_n = W_topk <= self.competition_thresh * W_min #size:(subbatch_size,Kw,fh*fw)
                mask = (p_mask * w_mask_p + n_mask*w_mask_n) > 0 #size:(subbatch_size,Kw,fh*fw)

                masked_additions = patches_batch*mask.float()

                delta.index_add_(0,U_inds_flat,masked_additions.view(-1,masked_additions.size()[-1]))

            delta = delta.view(delta.size()[0],delta.size()[1],1,1)
            
        else:
            


            #OLD IMPLEMENTATION:
            
            print(U.size(),U_inds.size(),patches.size())

            # W_top = torch.gather(W, dim=1,index=U_inds)
            raise RuntimeError()
            
            #Create masks for negative and positive values in each patch 
            mask_pos = patches > 0
            mask_neg = patches < 0

            filter_w, filter_h = self.filter_size

            

            #Initialize the update to the filters 
            delta = Variable(torch.zeros(W.size()),requires_grad=False,volatile=True)
            act_width,act_height = U_inds.size()[2:]

            #Loop through post synaptic activations
            for r in range(act_height):
                for c in range(act_width):

                    #Grab the presynaptic patch,masks, and indicies of the most activated
                    # filters for the current post synaptic activation 
                    p_mask = mask_pos[:,:,r:r+1,c:c+1].unsqueeze(1)
                    n_mask = mask_neg[:,:,r:r+1,c:c+1].unsqueeze(1)
                    x_patch  = patches[:,:,r:r+1,c:c+1] #size:(N, d, 1,1)
                    top_inds = U_inds[:,:,r,c] #size:(N, Kw)

                    #Slice out the weights of the top filters  
                    w_top_k = torch.cat([torch.index_select(W, 0, ti).unsqueeze(0) for ti in top_inds],0)

                    #Create a mask for filter weights that should be updated
                    w_mask_p = w_top_k >= self.competition_thresh * w_top_k[:,0].unsqueeze(1)
                    w_mask_n = w_top_k <= self.competition_thresh * w_top_k[:,-1].unsqueeze(1) 
                    mask = (p_mask * w_mask_p + n_mask*w_mask_n) > 0

                    #Apply the mask to the patch
                    temp = mask.float()*x_patch.unsqueeze(1)

                    #Add each patch to the correct place in delta
                    for t,inds in zip(temp,top_inds):
                        delta.index_add_(0, inds, t)
                    
        #Apply the update
        self.conv.weight.data += .001* delta.data

        #Normalize each filter
        last = W.size()[1:]
        as_lin_trans = self.conv.weight.squeeze()
        self.conv.weight.data = F.normalize(as_lin_trans,dim=1).view(-1,*last).data
        
        return delta



    def update_bias(self,a):
        self._assert_NHWC(a)
        if(not self.initialized): self.initialize()
        current = torch.mean(torch.mean(torch.mean(torch.sign(a),dim=0),dim=0),dim=0)
        self.running_avg = 0.95*self.running_avg + .05*current        
        self.bias += .05*(self.running_avg - self.average_activation)


