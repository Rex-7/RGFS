import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Add parent directory to sys.path to ensure that the feature_env module located in the upper level can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_env import FeatureEvaluator
from utils.logger import info

SOS_ID = -1
EOS_ID = -1


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float("inf"))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(
            batch_size, -1, source_len
        )

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(
            self.output_proj(combined.view(-1, self.input_dim + self.source_dim))
        ).view(batch_size, -1, self.output_dim)

        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = "attention_score"
    KEY_LENGTH = "length"
    KEY_SEQUENCE = "sequence"

    def __init__(self, layers, vocab_size, hidden_size, dropout, length, gpu):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length  # total length to decode
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        # In 0/1 vocabulary: 0 = feature not selected, 1 = feature selected
        self.sos_id = 0  # SOS: feature not selected
        self.eos_id = 1  # EOS: feature selected (actually no special EOS needed)
        self.gpu = gpu


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerDecoder(Decoder):
    def __init__(
        self,
        num_decoder_layers,
        nhead,
        vocab_size,
        embedding_size,
        dropout,
        activation,
        dim_feedforward,
        batch_first,
        length,
        gpu,
        latent_dim=None,
        max_seq_len=5000
    ):  # New: latent space dimension
        super(TransformerDecoder, self).__init__(
            num_decoder_layers, vocab_size, embedding_size, dropout, length, gpu
        )
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim if latent_dim is not None else embedding_size

        # Latent space to decoder dimension projection layer (e* to decoder input in the paper)
        if self.latent_dim != embedding_size:
            self.memory_proj = nn.Linear(self.latent_dim, embedding_size)
        else:
            self.memory_proj = nn.Identity()

        # positional layer - max_len should be based on sequence length, not vocabulary size
        self.positionalEncoding = PositionalEncoding(
            d_model=embedding_size, dropout=dropout, max_len=max_seq_len
        )  # Use a large enough value to support various sequence lengths
        # decoder layer
        self.decoderLayer = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
        )
        # stack decoder layer to construct transformer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoderLayer, num_layers=num_decoder_layers
        )
        self.attention = Attention(embedding_size)
        # output - modified to binary classification (0/1)
        self.out = nn.Linear(embedding_size, 2)
        # # out put
        # self.out = nn.Linear(embedding_size, vocab_size)

    def forward_train_valid(self, x, z):
        """
        Training/validation forward pass
        Args:
            x: target sequence [batch, seq]
            z: latent representation (e*) [batch, seq, latent_dim]
        """
        batch_size = x.shape[0]
        output_size = x.shape[1]
        x = x.cuda(self.gpu)
        embedded = self.embedding(x)
        embedded = self.positionalEncoding(embedded)

        # Project z to decoder dimension (latent_dim -> embedding_size)
        memory = self.memory_proj(z)

        # Restore causal mask to ensure sequential dependency of feature selection
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(output_size).cuda(
            self.gpu
        )
        # In feature selection, sequential constraints are needed to avoid redundant selection
        out = self.decoder(embedded, memory, tgt_mask)

        # Output 0/1 probability distribution (batch_size, seq_len, 2)
        predict_softmax = F.log_softmax(
            self.out(out.contiguous().view(-1, self.embedding_size)), dim=1
        )
        predict_softmax = predict_softmax.view(batch_size, output_size, 2)

        return predict_softmax

    def forward_step(self, z, input_id):
        """Single-step decoding"""
        embedded = self.embedding(input_id)
        embedded = self.positionalEncoding(embedded)

        # Project z to decoder dimension
        memory = self.memory_proj(z)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            input_id.shape[1]
        ).cuda(self.gpu)
        out = self.decoder(embedded, memory, tgt_mask)

        # Output 0/1 prediction (only take the prediction at the last position)
        predict_softmax = F.log_softmax(
            self.out(out.contiguous().view(-1, self.embedding_size)), dim=1
        )
        _, next_input_id = predict_softmax.max(dim=1, keepdim=True)
        output_id = next_input_id.reshape(input_id.shape[0], input_id.shape[1])
        return output_id

    def forward_infer(self, z):
        """
        Generate 0/1 feature selection sequence

        Now each position predicts 0 (not selected) or 1 (selected) for the feature at that position.
        Sequence length equals to the number of features.

        Args:
            z: latent representation [batch, seq, latent_dim]
        """
        batch_size = z.shape[0]
        input_id = (
            torch.LongTensor([self.sos_id] * batch_size)
            .view(batch_size, 1)
            .cuda(self.gpu)
        )
        generated_sequence = []

        for step in range(self.length):
            output_id = self.forward_step(z, input_id)
            # Save the prediction of this step
            generated_sequence.append(output_id[:, -1:])  # Only take the latest predicted token
            input_id = torch.cat((input_id, output_id[:, -1].reshape(-1, 1)), dim=1)

        # Concatenate all predictions to build the complete 0/1 sequence [batch_size, self.length]
        complete_sequence = torch.cat(generated_sequence, dim=1)
        return complete_sequence


def construct_decoder(fe: FeatureEvaluator, args) -> Decoder:
    """Construct TransformerDecoder"""
    size = fe.ds_size
    info(f"Construct TransformerDecoder...")

    return TransformerDecoder(
        num_decoder_layers=args.transformer_decoder_layers,
        nhead=args.decoder_nhead,
        vocab_size=2,  # 0 (not selected) or 1 (selected)
        embedding_size=args.decoder_embedding_size,
        dropout=args.transformer_decoder_dropout,
        activation=args.transformer_decoder_activation,
        dim_feedforward=args.decoder_dim_feedforward,
        batch_first=args.batch_first,
        length=size,
        gpu=args.gpu,
        latent_dim=args.d_latent_dim,  # New: passing latent space dimension
        max_seq_len=args.max_seq_len
    )
