# controller.py
import torch
import torch.nn as nn
import os
import sys

# Add parent directory to sys.path to ensure that the feature_env module located in the upper level can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from decoder import construct_decoder
from encoder import construct_encoder, TabEncoder
from feature_env import FeatureEvaluator

SOS_ID = 0
EOS_ID = 0


# gradient based automatic feature selection
class GAFS(nn.Module):
    def __init__(self,
                 fe:FeatureEvaluator,
                 args
                 ):
        super(GAFS, self).__init__()
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args)
        self.decoder = construct_decoder(fe, args)
        
        # Add TabEncoder support - consistent with DIFFT
        if hasattr(args, 'tab_len') and args.tab_len > 0:
            self.tab_encoder = TabEncoder(
                input_size=args.tab_len, 
                hidden_size=args.encoder_embedding_size, 
                dropout=args.transformer_encoder_dropout, 
                num_layers=4
            )
        else:
            self.tab_encoder = None
    
    def set_sparse_mask(self, mask_generator):
        """
        Set structured sparse attention mask to Encoder
        
        Args:
            mask_generator: FeatureMaskGenerator instance
        """
        device = f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu'
        
        # Set mask for Encoder only (boolean mask)
        encoder_mask = mask_generator.get_encoder_mask(device=device)
        self.encoder.set_sparse_mask(encoder_mask)
        
        # Decoder remains as is, no sparse mask applied

    def forward(self, input_variable, target_variable=None, tab=None):
        """
        GAFS forward pass - supports tabular data as additional input
        Args:
            input_variable: input sequence
            target_variable: target sequence
            tab: tabular data (optional) - consistent with DIFFT
        Returns:
            If tab: predict_value, decoder_outputs, feat, mu, logvar, tab_emb
            If no tab: predict_value, decoder_outputs, feat, mu, logvar
        """
        # TransformerVAE encoder - now returns z (sampled latent vector)
        z, predict_value, mu, logvar = self.encoder.forward(input_variable)
        # TransformerDecoder - uses z as memory for decoder
        decoder_outputs = self.decoder.forward_train_valid(target_variable, z)
        _, feat = decoder_outputs.max(2, keepdim=True)
        feat = feat.reshape(input_variable.size(0), input_variable.size(1))

        # Process tabular data - consistent with DIFFT TransformerVAE
        if tab is not None and self.tab_encoder is not None:
            tab_emb = self.tab_encoder(tab)
            return predict_value, decoder_outputs, feat, mu, logvar, tab_emb
        else:
            return predict_value, decoder_outputs, feat, mu, logvar


    def generate_new_feature(self, input_variable, predict_lambda=1, direction='-'):
        """TransformerVAE generates new features"""
        encoder_outputs, predict_value, new_encoder_outputs = \
            self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_feat_seq = self.decoder.forward_infer(new_encoder_outputs)
        
        return new_feat_seq
    
    def encode(self, input_variable, tab=None):
        """
        Encode input sequence into latent space - consistent with DIFFT interface
        Args:
            input_variable: input sequence
            tab: tabular data (optional)
        Returns:
            If tab: z, mu, logvar, predict_value, tab_emb
            If no tab: z, mu, logvar, predict_value
        """
        # encoder now directly returns the sampled z
        z, predict_value, mu, logvar = self.encoder.forward(input_variable)
        
        # Process tabular data
        if tab is not None and self.tab_encoder is not None:
            tab_emb = self.tab_encoder(tab)
            return z, mu, logvar, predict_value, tab_emb
        else:
            return z, mu, logvar, predict_value

    def evaluate(self, z):
        """
        General evaluation function: supports arbitrary feature length and Batch size
        """
        if len(z.shape) == 3:
            # z shape is [Batch, Seq, Dim]
            # Regardless of Seq length, directly average pool across the Seq dimension (dim=1)
            # This yields a global feature vector [Batch, Dim] for each sample
            z = z.mean(dim=1)

        elif len(z.shape) == 2:
            # Already in [Batch, Dim] format, no processing needed
            pass
        else:
            raise ValueError(
                f"Unexpected z shape: {z.shape}. Expected 2D or 3D tensor."
            )

        # Input to MLP is always [Batch, Dim]
        out = self.encoder.mlp(z)
        out = self.encoder.regressor(out)
        evaluation = torch.sigmoid(out)  # Result is [Batch, 1]

        return evaluation

    def generate(self, z, max_len=None):
        """
        Generate sequence from LDM latent space - refer to DIFFT's TransformerVAE.generate implementation
        Args:
            z: latent representation [seq_len, batch_size, latent_dim] or [batch_size, seq_len, latent_dim]
            max_len: maximum generation length (optional)
        Returns:
            generated_seq: generated feature sequence [batch_size, seq_len]
        """
        # Process dimensions: ensure z is in [batch_size, seq_len, latent_dim] format
        if len(z.shape) == 3:
            if z.shape[0] > z.shape[1]:
                # [seq_len, batch_size, latent_dim] -> [batch_size, seq_len, latent_dim]
                z = z.permute(1, 0, 2).contiguous()
            # Otherwise already [batch_size, seq_len, latent_dim]
        
        # Use decoder's inference method to generate sequence
        generated_seq = self.decoder.forward_infer(z)
        
        return generated_seq

