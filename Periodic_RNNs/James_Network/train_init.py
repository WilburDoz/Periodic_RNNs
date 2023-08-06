import torch
import parameters_will
import torch.optim as optim
import RNN_Will as _model_
import numpy as np
import pickle
import utils
from datetime import datetime
import os

# Set up our parameters
params = parameters_will.default_params()

print_iters = 100
save_iters = 10000

# make instance of model
model = _model_.ReInit_RNN(params.model)
# put model to gpu (if available)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Make an ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr=params.train.learning_rate, weight_decay=params.train.weight_decay)
min_loss = np.infty

# Setup save directory
path = os.getcwd()
now = datetime.now()
current_time = now.strftime("%y_%m_%d_%H%M%S")
directory = path + "/" + current_time + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

for train_i in range(params.train.train_iters):

    # 1. Get input data, and convert to tensors (I have assumed you will put inputs etc into a dictionary)
    input_dict = utils.generate_data_easy(params)#, freqs = np.full(params.data.batch_size, 7, dtype=int))
    
    # set all gradients to None
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None
        
    # forward pass
    variables = model(input_dict, device=device)
    
    # collate inputs for model
    (losses, loss_fit) = _model_.compute_losses_torch(input_dict, variables, model, params.train, device=device)

    # backward pass
    losses.backward()

    # clip gradients (you don't have to do this but it's a good idea for RNNs)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    optimizer.step()
    
    if losses.detach() < min_loss:
        min_loss = losses.detach()
        best_model = model
        print(f"{train_i}, new PB! {min_loss}")

    if train_i % print_iters == 0:
        print(f"{train_i}, {losses.item():.5f}, {loss_fit.item():.5f}, {losses.item()-loss_fit.item():.5f}")
        
    if train_i % save_iters == 0:
        model_name = directory + str(train_i)
        torch.save(model, model_name)
        