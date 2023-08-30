#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import parameters_will


class VanillaRNN(nn.Module):
    def __init__(self, par):
        super(VanillaRNN, self).__init__()

        self.par = par
        self.batch_size = par.batch_size
        #self.seq_len = par.seq_len
        self.device = None

        # RNN Activation
        if self.par.hidden_act == 'none':
            self.activation = nn.Identity()
        elif self.par.hidden_act == 'relu':
            self.activation = nn.ReLU()
        elif self.par.hidden_act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.par.hidden_act == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Activation ' + str(self.par.hidden_act) + ' not implemented yet')
            
        # RNN Activation
        if self.par.output_act == 'none':
            self.out_activation = nn.Identity()
        elif self.par.output_act == 'relu':
            self.out_activation = nn.ReLU()
        elif self.par.output_act == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif self.par.output_act == 'tanh':
            self.out_activation = nn.Tanh()
        else:
            raise ValueError('Activation ' + str(self.par.output_act) + ' not implemented yet')

        # RNN Init
        self.hidden_init = nn.Parameter(torch.zeros((1, self.par.h_size), dtype=torch.float32),
                                        requires_grad=self.par.hidden_init_learn)

        # RNN Transition
        self.transition = nn.Linear(self.par.h_size, self.par.h_size, bias=True)

        # Input embedding
        self.embedding = nn.Linear(self.par.i_size, self.par.h_size, bias=True)

        # Predictions
        self.predict = nn.Linear(self.par.h_size, self.par.t_size, bias=True)  # add activation of your choice here

        _ = self.apply(self._init_weights)

    def _init_weights(self, module):
        # choose how to initialise weights
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.linear_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.embedding_std)

        nn.init.normal_(self.hidden_init, mean=0.0, std=self.par.hidden_init_std)

        if self.par.transition_init == 'orthogonal':
            nn.init.orthogonal_(self.transition.weight)
        elif self.par.transition_init == 'identity':
            nn.init.eye_(self.transition.weight)
        nn.init.constant_(self.transition.bias, 0)

    def forward_old(self, inputs, device='cpu'):
        # embed inputs
        #input_to_hidden = inputs.observation.reshape((-1, self.par.i_size)).reshape((self.seq_len, self.batch_size, -1))
        input_to_hidden = inputs.observation[:,:,None]
        
        # initialise hidden
        h = self.activation(self.hidden_init.tile([self.batch_size, 1]))  # (can remove activation here)
        pred = self.predict(h)
        hs, preds = [], []

        # Run RNN
        for i, (i_to_h) in enumerate(input_to_hidden):
            # this is your thing to gate the input (you need to specify inputs.input_or_rnn which is batch x seq_len x 1
            input_or_rnn = inputs.input_or_rnn[:,i] * i_to_h + (1.0 - inputs.input_or_rnn[:,i]) * pred
            # path integrate (generative)
            h = self.activation(self.transition(h) + input_or_rnn)
            # store rnn hidden states
            hs.append(h)
            
            # prediction (this makes a prediction for next time-step (can add activation post transition if you want))
            pred = self.out_activation(self.predict(h))
            # store rnn hidden states
            preds.append(pred)

        # collect variables
        variable_dict = parameters_will.DotDict(
            {'hidden': hs,
             'pred': preds,
             })

        return variable_dict
    
    def forward(self, inputs, device='cpu'):
        
        # initialise hidden
        h = self.activation(self.hidden_init.tile([self.batch_size, 1]))  # (can remove activation here)
        pred = self.predict(h)
        hs, preds, preactivations = [], [], []

        # Run RNN
        for i, (i_to_h) in enumerate(inputs.observation):
            # this is your thing to gate the input (you need to specify inputs.input_or_rnn which is batch x seq_len x 1
            hidden_input = inputs.input_or_rnn[i,:][:,None] * self.embedding(i_to_h[:,None]) + (1.0 - inputs.input_or_rnn[i,:][:,None]) *  self.embedding(pred)
            preactivation = self.transition(h) + hidden_input
            # path integrate (generative)
            h = self.activation(preactivation)
            # store rnn hidden states
            hs.append(h)
            preactivations.append(preactivation)
            
            # prediction (this makes a prediction for next time-step (can add activation post transition if you want))
            pred = self.out_activation(self.predict(h))
            # store rnn hidden states
            preds.append(pred)

        # collect variables
        variable_dict = parameters_will.DotDict(
            {'hidden': hs,
             'pred': preds,
             'preactivations':preactivations
             })

        return variable_dict
    
class ReInit_RNN(nn.Module):
    def __init__(self, par):
        super(ReInit_RNN, self).__init__()

        self.par = par
        self.batch_size = par.batch_size
        #self.seq_len = par.seq_len
        self.device = None

        # RNN Activation
        if self.par.hidden_act == 'none':
            self.activation = nn.Identity()
        elif self.par.hidden_act == 'relu':
            self.activation = nn.ReLU()
        elif self.par.hidden_act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.par.hidden_act == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Activation ' + str(self.par.hidden_act) + ' not implemented yet')
            
        # RNN Activation
        if self.par.output_act == 'none':
            self.out_activation = nn.Identity()
        elif self.par.output_act == 'relu':
            self.out_activation = nn.ReLU()
        elif self.par.output_act == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        elif self.par.output_act == 'tanh':
            self.out_activation = nn.Tanh()
        else:
            raise ValueError('Activation ' + str(self.par.output_act) + ' not implemented yet')

        # RNN Init
        self.hidden_init = nn.Parameter(torch.zeros((self.par.num_inits, self.par.h_size), dtype=torch.float32),
                                        requires_grad=self.par.hidden_init_learn)

        # RNN Transition
        self.transition = nn.Linear(self.par.h_size, self.par.h_size, bias=True)

        # Input embedding
        self.embedding = nn.Linear(self.par.i_size, self.par.h_size, bias=True)

        # Predictions
        self.predict = nn.Linear(self.par.h_size, self.par.t_size, bias=True)  # add activation of your choice here

        _ = self.apply(self._init_weights)

    def _init_weights(self, module):
        # choose how to initialise weights
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.linear_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.embedding_std)

        nn.init.normal_(self.hidden_init, mean=0.0, std=self.par.hidden_init_std)

        if self.par.transition_init == 'orthogonal':
            nn.init.orthogonal_(self.transition.weight)
        elif self.par.transition_init == 'identity':
            nn.init.eye_(self.transition.weight)
        nn.init.constant_(self.transition.bias, 0)
    
    def forward(self, inputs, device='cpu'):
        
        # initialise hidden
        h = self.activation(self.hidden_init[inputs.freqs, :])  # (can remove activation here)
        pred = self.predict(h)
        hs, preds = [], []

        # Run RNN
        for i, (i_to_h) in enumerate(inputs.outputs):
            # this is your thing to gate the input (you need to specify inputs.input_or_rnn which is batch x seq_len x 1
            hidden_input = self.embedding(pred)
            # path integrate (generative)
            h = self.activation(self.transition(h) + hidden_input)
            # store rnn hidden states
            hs.append(h)
            
            # prediction (this makes a prediction for next time-step (can add activation post transition if you want))
            pred = self.out_activation(self.predict(h))
            # store rnn hidden states
            preds.append(pred)

        # collect variables
        variable_dict = parameters_will.DotDict(
            {'hidden': hs,
             'pred': preds,
             })

        return variable_dict
    


def compute_losses_torch(model_in, model_out, model, par, device='cpu'):
    loss_fit = torch.sum(torch.pow(model_in.outputs[:,:,None] - torch.stack(model_out.pred), 2))
    loss_act = torch.sum(torch.pow(torch.stack(model_out.hidden), 2))
    return (loss_fit + par.act_weight*loss_act, loss_fit)
