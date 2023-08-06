#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import parameters_will
import torch.optim as optim
import RNN_Will as _model_

# Initialise hyper-parameters for model
params = parameters_will.default_params()

# make instance of model
model = _model_.VanillaRNN(params.model)
# put model to gpu (if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Make an ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=params.train.learning_rate, weight_decay=params.train.weight_decay)

for train_i in range(params.train.train_iters):

    # 1. Get input data, and convert to tensors (I have assumed you will put inputs etc into a dictionary)
    """
    inputs_torch = your code here
    """

    # set all gradients to None
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None
    # forward pass
    variables = model(inputs_torch, device=device)
    # collate inputs for model
    losses = _model_.compute_losses_torch(inputs_torch, variables, model, params.train, device=device)
    # backward pass
    losses.train_loss.backward()
    # clip gradients (you don't have to do this but it's a good idea for RNNs)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
    optimizer.step()
