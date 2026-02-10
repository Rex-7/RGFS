# encoder.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from feature_env import FeatureEvaluator
from utils.logger import info


class TabEncoder(nn.Module):
    """Tabular data encoder - consistent with DIFFT fly"""
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super(TabEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        current_size = input_size
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_size, hidden_size))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            current_size = hidden_size

        self.final_layer = nn.Linear(hidden_size, hidden_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norm_layers):
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual.shape == x.shape:  # Add residual only when dimensions match
                x = x + residual
            x = norm(x)

        x = self.final_layer(x)
        x = self.final_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, predict_value, mu, logvar = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        return encoder_outputs, predict_value, new_encoder_outputs

    def forward(self, x):
        pass

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Initialize PE (positional encoding) with shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # Initialize a tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # This is the content inside sin and cos brackets, transformed via e and ln
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # Calculate PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Calculate PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Unsqueeze a batch dimension on the outside for convenience
        pe = pe.unsqueeze(0)
        # Use register_buffer if a parameter doesn't participate in gradient descent but should be saved with the model
        # This way it can be saved via register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x is the inputs after embedding, e.g., (1, 7, 128): batch size 1, 7 words, word dimension 128
        """
        # Add x and positional encoding.
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerEncoderVAE(Encoder):
    def __init__(
            self,
            num_encoder_layers,
            nhead, 
            vocab_size,
            embedding_size,
            dropout, 
            activation,
            dim_feedforward,
            batch_first,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
            d_latent_dim,
            max_seq_len=5000
            ):
        super(TransformerEncoderVAE, self).__init__(num_encoder_layers, vocab_size, embedding_size)
        # positional layer - max_len should be based on sequence length, not vocabulary size
        self.positionalEncoding = PositionalEncoding(
                                d_model = embedding_size,
                                dropout = dropout,
                                max_len = max_seq_len)  # Use a large enough value to support various sequence lengths
        # multi-head attention && feed forward && norm -> encoder layer 
        self.encoderLayer = nn.TransformerEncoderLayer(
                                d_model = embedding_size,
                                nhead = nhead,
                                dropout = dropout,
                                activation = activation,
                                dim_feedforward = dim_feedforward,
                                batch_first = batch_first)
        # stack encoder layers to construct transformer encoder
        self.encoder = nn.TransformerEncoder(
                                encoder_layer = self.encoderLayer,
                                num_layers = num_encoder_layers)
        self.mu = nn.Linear(embedding_size, d_latent_dim)
        self.logvar = nn.Linear(embedding_size, d_latent_dim)
        
        # Structured sparse attention mask
        self.sparse_mask = None
        # mlp layer
        self.mlp = nn.Sequential()
        for i in range(mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(d_latent_dim, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(mlp_hidden_size, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(d_latent_dim if mlp_layers == 0 else mlp_hidden_size, 1)
    
    def set_sparse_mask(self, mask):
        """
        Set structured sparse attention mask
        
        Args:
            mask: torch tensor, shape [seq_len, seq_len], boolean mask (True=masked)
        """
        self.sparse_mask = mask
        info(f'Encoder: Sparse mask set, shape={mask.shape}, sparsity={(mask.sum() / mask.numel()):.2%}')
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # get embedding
        embedding = self.embedding(x)
        # add positional information
        embedding = self.positionalEncoding(embedding)
        
        # encoder output: [batch, seq, hidden]
        # Apply sparse mask to encoder if set
        if self.sparse_mask is not None:
            out = self.encoder(embedding, mask=self.sparse_mask)
        else:
            out = self.encoder(embedding)
        
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        
        # VAE: Calculate mu and logvar for each position (Paper Eq. 2)
        # mu, logvar shape: [batch, seq, latent]
        mu = self.mu(out)
        logvar = self.logvar(out)
        
        # Reparameterization sampling: z = mu + eps * exp(logvar/2)
        # z shape: [batch, seq, latent]
        z = self.reparameterize(mu, logvar)
        
        # evaluator: evaluate after taking sequence average of z
        z_mean = z.mean(dim=1)  # [batch, latent]
        if self.mlp is not None and self.regressor is not None:
            out = self.mlp(z_mean)
            out = self.regressor(out)
            predict_value = torch.sigmoid(out)
        else:
            predict_value = torch.zeros(z.shape[0], 1, device=z.device)

        # Return sampled z (e* in the paper), instead of original encoder_outputs
        # z shape: [batch, seq, latent]
        return z, predict_value, mu, logvar


def construct_encoder(fe: FeatureEvaluator, args) -> Encoder:
    """Construct TransformerEncoderVAE encoder"""
    size = fe.ds_size
    info(f'Construct TransformerEncoderVAE Encoder...')
    
    return TransformerEncoderVAE(
        num_encoder_layers = args.transformer_encoder_layers,
        nhead = args.encoder_nhead,
        vocab_size = 2,  # two tokens 0 and 1 represent feature selection
        embedding_size = args.encoder_embedding_size,
        dropout = args.transformer_encoder_dropout,
        activation = args.transformer_encoder_activation,
        dim_feedforward = args.encoder_dim_feedforward,
        batch_first = args.batch_first,
        mlp_layers = args.mlp_layers,
        mlp_hidden_size = args.mlp_hidden_size,
        mlp_dropout = args.encoder_dropout,
        d_latent_dim = args.d_latent_dim,
        max_seq_len=args.max_seq_len
    )
