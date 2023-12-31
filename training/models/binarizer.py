# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities

import torch
from torch import autograd
import math


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold
        # %
        threshold = torch.sigmoid(threshold).item()

        mask = inputs.clone()
        
        _, idx = inputs.flatten().sort(descending=True)
        j = math.ceil(threshold * inputs.numel())

        #flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0.
        flat_out[idx[:j]] = 1.
        
        ctx.save_for_backward(mask)  # we should try two things

        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        mask, = ctx.saved_tensors
        return gradOutput, ((gradOutput * mask).sum()).view(-1)
