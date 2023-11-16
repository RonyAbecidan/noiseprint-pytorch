import torch
import torch.nn as nn
from utilityRead import *
import torchvision.transforms as transforms

#torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)

class AddBias(nn.Module):
    """A customized layer to add bias"""
    def __init__(self, num_features):
        super(AddBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, bnorm_init_gamma, bnorm_init_var, bnorm_decay, bnorm_epsilon):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.bnorm_decay = bnorm_decay
        self.bnorm_epsilon = bnorm_epsilon

        self.gamma=nn.Parameter(torch.ones(num_features))
        self.moving_mean=nn.Parameter(torch.ones(num_features))
        self.moving_variance=nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        _, C, _, _ = x.shape

        x = (x - self.moving_mean.reshape((1,C,1,1))) / torch.pow(self.moving_variance.reshape((1,C,1,1)) + self.bnorm_epsilon,0.5)
        return self.gamma.reshape((1,C,1,1))*x

class FullConvNet(nn.Module):
    def __init__(self, bnorm_decay, flag_train, num_levels=17):
        super(FullConvNet, self).__init__()

        self._num_levels = num_levels
        self._actfun = [nn.ReLU(), ] * (self._num_levels-1) + [nn.Identity(), ]
        self._f_size = [3, ] * self._num_levels
        self._f_num_in = [1, ] + [64, ] * (self._num_levels-1)
        self._f_num_out = [64, ] * (self._num_levels-1) + [1, ]
        self._f_stride = [1, ] * self._num_levels
        self._bnorm = [False, ] + [True, ]*(self._num_levels-2) + [False, ]
        self.conv_bias = [True, ] + [False, ]*(self._num_levels-2) + [True, ]
        self._bnorm_init_var = 1e-4
        self._bnorm_init_gamma = torch.sqrt(torch.tensor(2.0/(9.0*64.0)))
        self._bnorm_epsilon = 1e-5
        self._bnorm_decay = bnorm_decay

        self.level = [None, ] * self._num_levels
        self.flag_train = flag_train
        self.extra_train = []
        self.conv_layers = nn.ModuleList()

        for i in range(self._num_levels):
            self.conv_layers.append(self._conv_layer(self._f_size[i],self._f_num_in[i],self._f_num_out[i],self._f_stride[i], self._bnorm[i], self.conv_bias[i],self._actfun[i]))

    def forward(self, x):
        for i in range(self._num_levels):
            x = self.conv_layers[i](x)
            self.level[i] = x

        return x

    def _batch_norm(self,out_filters):
        batch_norm = CustomBatchNorm(out_filters, self._bnorm_init_gamma, self._bnorm_init_var, self._bnorm_decay, self._bnorm_epsilon)
        return batch_norm

    def _conv_layer(self,filter_size,in_filters,out_filters, stride, apply_bnorm,conv_bias,actfun):
        # Calcul du padding nécessaire pour le 'same' padding
        layers = []
        layers.append(nn.Conv2d(in_channels=in_filters,out_channels=out_filters, kernel_size=filter_size, stride=stride, padding="same",bias=conv_bias))

        if apply_bnorm:
            layers.append(self._batch_norm(out_filters))

        #if the bias was not already added with Conv2d, we add it manually
        if not(conv_bias):
            layers.append(AddBias(out_filters))
        
        layers.append(actfun)  # activation function 
        
        return nn.Sequential(*layers)


def getNoiseprint(image_path):

    '''
        #
    # Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
    # All rights reserved.
    # This work should only be used for nonprofit purposes.
    #
    # By downloading and/or using any of these files, you implicitly agree to all the
    # terms of the license, as specified in the document LICENSE.txt
    # (included in this package) and online at
    # http://www.grip.unina.it/download/LICENSE_OPEN.txt
    #
    """
    @author: davide.cozzolino
    """
    '''

    img,mode = imread2f(image_path, channel=1)
    
    slide = 1024 #3072
    largeLimit = 1050000 #9437184
    overlap = 34
    transform = transforms.ToTensor()
    
    try:
        QF = jpeg_qtableinv(image_path)
    except:
        QF = 101
    
    net = FullConvNet(0.9, torch.tensor(False), num_levels=17)
    file_path=f"pretrained_weights/model_qf{int(QF)}.pth"
    net.load_state_dict(torch.load(file_path))
    net.eval()
    
    with torch.no_grad():
    
        if img.shape[0]*img.shape[1]>largeLimit:
            print(' %dx%d large %3d' % (img.shape[0], img.shape[1], QF))
            # for large image the network is executed windows with partial overlapping 
            res = np.zeros((img.shape[0],img.shape[1]), np.float32)
            for index0 in range(0,img.shape[0],slide):
                index0start = index0-overlap
                index0end   = index0+slide+overlap
                
                for index1 in range(0,img.shape[1],slide):
                    index1start = index1-overlap
                    index1end   = index1+slide+overlap
                    clip = img[max(index0start, 0): min(index0end,  img.shape[0]), \
                               max(index1start, 0): min(index1end,  img.shape[1])]
        
                    tensor_image = transform(clip)
                    tensor_image = tensor_image.reshape(1,1,tensor_image.shape[1],tensor_image.shape[2])
                    resB = net(tensor_image)
        
                    
                    resB = resB[0][0]
        
                    if index0>0:
                        resB = resB[overlap:, :]
                    if index1>0:
                        resB = resB[:, overlap:]
                    resB = resB[:min(slide,resB.shape[0]), :min(slide,resB.shape[1])]
                    
                    res[index0: min(index0+slide,  res.shape[0]), \
                        index1: min(index1+slide,  res.shape[1])]  = resB


        else:
            print(' %dx%d small %3d' % (img.shape[0], img.shape[1], QF))
            tensor_image = transform(img)
            tensor_image = tensor_image.reshape(1,1,tensor_image.shape[1],tensor_image.shape[2])
            res = net(tensor_image)
            res = (res[0][0]).numpy()
            
    return img,res