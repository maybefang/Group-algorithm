# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import TopKBinarizer

# This function is from
# https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays


class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask during training,
    and does real pruning during inference
    mask should mark the pruned col
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        pruning: bool = True
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Default: ``topK``
            
            bias_mask:
                Prune bias or not
                Default: False
            pruning:
                Do Pruning or not
                Default: True
        """
        super(
            MaskedLinear,
            self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)   #if bias=True, self.bias is the tensor
        
        
        self.pruning = pruning     #bool

        self.inference_mode = False
        
        self.mask_scores = None  #存疑自己训练还是将weight+grad作为scores
        
        self.threshold = nn.Parameter(torch.zeros(1) + 10.0)

        self.mask = nn.Parameter(torch.ones(in_features))
        

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound) 
    #     with torch.no_grad():
    #         #std = self.weight.std()
    #         self.threshold.data.fill_(10.)

    def get_mask(self):
        # get head mask
        if self.weight.grad is None:
            self.mask_scores = torch.abs(torch.sum(self.weight.data, dim=0)/float(self.weight.shape[0]))
        else:
            abs_weight = torch.abs(self.weight + self.weight.grad)
            self.mask_scores = torch.sum(abs_weight.data, dim=0)/float(abs_weight.shape[0]) #mask_scores是每一列(weight+grad)绝对值的均值
        
        if self.pruning:
            mask = TopKBinarizer.apply(
                self.mask_scores, self.threshold)
        else:
            mask = torch.ones()
        self.mask = nn.Parameter(mask, requires_grad=False)
        return mask

    def make_inference_pruning(self, blocksize):
        self.inference_mode = True
        weight_shape = self.weight.size()
        
        #self.mask = self.get_mask()
        
        if not self.pruning:
            mask = torch.ones_like(self.weight[0])

        self.mask = self.mask.type('torch.BoolTensor').view(-1)
        
        self.weight = nn.Parameter(self.weight[:, mask])  #left col where mask=1
        if self.bias_mask:
            self.bias = nn.Parameter(self.bias[mask])

        # we do not need those parameters!
        self.mask_scores = None
        self.threshold = None
        # we need this mask for some Layer O and FC2 pruning
        return mask

    
    def forward(self, input: torch.tensor):
        if not self.inference_mode:
            output = self.training_forward(input)
        else:
            output = self.inference_forward(input)
        return output


    def inference_forward(self, input: torch.tensor):
        return F.linear(input, self.weight, self.bias)

    def training_forward(self, input: torch.tensor):
        mask = self.get_mask()

        weight_shape = self.weight.size()
        
        weight_thresholded = self.weight
        
        # Mask weights with computed mask
        if self.pruning:
            weight_thresholded = mask * weight_thresholded
            
        return F.linear(input, weight_thresholded, self.bias)



class MaskedConv2d(nn.Conv2d):
    """
    Fully Connected layer with on the fly adaptive mask during training,
    and does real pruning during inference
    mask should mark the pruned col
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        pruning: bool = True
    ):
        """
        Args:
            in_features (`int`)
                Size of each input sample
            out_features (`int`)
                Size of each output sample
            bias (`bool`)
                If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
            mask_init (`str`)
                The initialization method for the score matrix if a score matrix is needed.
                Choices: ["constant", "uniform", "kaiming"]
                Default: ``constant``
            mask_scale (`float`)
                The initialization parameter for the chosen initialization method `mask_init`.
                Default: ``0.``
            pruning_method (`str`)
                Method to compute the mask.
                Default: ``topK``
            
            bias_mask:
                Prune bias or not
                Default: False
            pruning:
                Do Pruning or not
                Default: True
        """
        super(
            MaskedConv2d,
            self).__init__(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride = stride,
            padding = padding,
            bias=bias)   #if bias=True, self.bias is the tensor
        # print("++++++++++++++++++++++++++++++++++++++")
        
        self.pruning = pruning     #bool

        self.inference_mode = False
        
        self.mask_scores = None

        self.mask = nn.Parameter(torch.ones(self.weight.shape[1],self.weight.shape[2],self.weight.shape[3]), requires_grad=False)
            
        self.threshold = nn.Parameter(torch.zeros(1) + 10.0)

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)
    #     with torch.no_grad():
    #         self.threshold.data.fill_(10.)
        
    def get_mask(self):
        weight_shape = self.weight.shape
        if self.weight.grad is None:
            self.mask_scores = torch.abs(torch.sum(self.weight.data, dim=0)/float(weight_shape[0]))
        else:
            abs_weight = torch.abs(self.weight + self.weight.grad)
            self.mask_scores = torch.sum(abs_weight.data, dim=0)/float(weight_shape[0]) #mask_scores是每一列(weight+grad)绝对值的均值
        if self.pruning:
            mask = TopKBinarizer.apply(
                self.mask_scores, self.threshold)
        else:
            mask = torch.ones()
        self.mask = nn.Parameter(mask, requires_grad=False)
        return mask

    def make_inference_pruning(self, blocksize):
        self.inference_mode = True
        weight_shape = self.weight.size()
        
        #self.mask = self.get_mask()
        
        if not self.pruning:
            mask = torch.ones_like(self.weight[0])

        self.mask = self.mask.type('torch.BoolTensor').view(-1)
        
        self.weight = nn.Parameter(self.weight[:, mask])  #left col where mask=1

        # we do not need those parameters!
        self.mask_scores = None
        self.threshold = None
        # we need this mask for some Layer O and FC2 pruning
        return mask

    
    def forward(self, input: torch.tensor):
        if not self.inference_mode:
            output = self.training_forward(input)
        else:
            output = self.inference_forward(input)
        return output


    def inference_forward(self, input: torch.tensor):
        return F.conv2d(input, self.weight, self.bias)

    def training_forward(self, input: torch.tensor):
        mask = self.get_mask()

        weight_shape = self.weight.size()
        
        weight_thresholded = self.weight
        
        # Mask weights with computed mask
        if self.pruning:
            weight_thresholded = mask * weight_thresholded
            
        return F.conv2d(input, weight_thresholded, self.bias, stride = self.stride, padding = self.padding)

