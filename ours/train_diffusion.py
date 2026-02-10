# train_diffusion.py
import argparse
import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from controller import GAFS
from diffusion_model import TransformerDM
from feature_env import FeatureEvaluator, base_path
from utils_meter import FSDataset, AvgrageMeter
from utils.logger import info, error
from sparse_mask import FeatureMaskGenerator  # Import mask generator
import time
from pipeline_diffusion import DiffTipeline
from utils.rlac_tools import downstream_task_new
from tqdm import tqdm
# Remove unused feature engineering related imports - now focusing on feature selection tasks
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import json

# Filter sklearn quantile warning
warnings.filterwarnings('ignore', message='n_quantiles .* is greater than the total number of samples')

# Add parent directory to sys.path to ensure that the feature_env module located in the upper level can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser()
# Base parameters
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task_name', type=str, default='openml_586')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method_name', type=str, default='transformerVae')

# VAE related parameters - consistent with train_controller.py
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=64)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=64)
parser.add_argument('--decoder_dropout', type=float, default=0)
parser.add_argument('--d_latent_dim', type=int, default=64)  # Maintain consistency with pretrained VAE model

# Transformer Encoder parameters - maintain consistency with pretrained model
parser.add_argument('--transformer_encoder_layers', type=int, default=2)
parser.add_argument('--encoder_nhead', type=int, default=8)
parser.add_argument('--encoder_embedding_size', type=int, default=64)  # Maintain consistency with pretrained model
parser.add_argument('--transformer_encoder_dropout', type=float, default=0.1)
parser.add_argument('--transformer_encoder_activation', type=str, default='relu')
parser.add_argument('--encoder_dim_feedforward', type=int, default=128)  # Maintain consistency with pretrained model

# Transformer Decoder parameters - maintain consistency with pretrained model
parser.add_argument('--transformer_decoder_layers', type=int, default=2)
parser.add_argument('--decoder_nhead', type=int, default=8)
parser.add_argument('--decoder_embedding_size', type=int, default=64)  # Maintain consistency with pretrained model
parser.add_argument('--transformer_decoder_dropout', type=float, default=0.1)
parser.add_argument('--transformer_decoder_activation', type=str, default='relu')
parser.add_argument('--decoder_dim_feedforward', type=int, default=128)  # Maintain consistency with pretrained model
parser.add_argument('--batch_first', type=bool, default=True)

# Diffusion Model parameters - consistent with DIFFT
parser.add_argument('--diff_hidden_size', type=int, default=512)
parser.add_argument('--diff_num_layers', type=int, default=8)
parser.add_argument('--diff_n_heads', type=int, default=8)
parser.add_argument('--diff_dropout', type=float, default=0.05)
parser.add_argument('--diff_timesteps', type=int, default=1000)
parser.add_argument('--diff_lr', type=float, default=0.0002)  # Adjust learning rate based on batch_size=8
parser.add_argument('--diff_epochs', type=int, default=800)  # Fix: Paper suggests 800 epochs

# TabEncoder parameters - consistent with DIFFT
parser.add_argument('--tab_len', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--accumulation_steps', type=int, default=16)
parser.add_argument('--prediction_type', type=str, default='epsilon', choices=['epsilon', 'sample', 'v_prediction'])
# Noise schedule parameters fully consistent with DIFFT
parser.add_argument('--beta_schedule', type=str, default='scaled_linear', choices=['linear', 'cosine', 'squaredcos_cap_v2', 'scaled_linear'])
parser.add_argument('--beta_start', type=float, default=0.00085)  # Consistent with DIFFT
parser.add_argument('--beta_end', type=float, default=0.012)  # Consistent with DIFFT
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--diff_num_step', type=int, default=50)  # Add missing parameter for inference steps
parser.add_argument('--guidance_scale', type=float, default=0.0)
parser.add_argument('--snr_gamma', type=float, default=5.0)  # Fix: Enable Min-SNR weighting strategy
parser.add_argument('--loss_type', type=str, default='mse')
parser.add_argument('--use_context', type=bool, default=True)

# Reward guidance parameters - key parameters from paper
parser.add_argument('--use_reward', type=float, default=100.0)  # Paper suggestion: reward guidance scale
parser.add_argument('--infer_method', type=str, default= 'RF')  # Paper suggestion: use Random Forest as downstream model

# Structured sparse attention parameters
parser.add_argument('--use_sparse_mask', action='store_true', default=False, help='Whether to use structured sparse attention mask')
parser.add_argument('--sparse_top_k', type=int, default=5, help='Number of neighbors to keep for each feature')
parser.add_argument('--sparse_method', type=str, default='correlation', help='Feature correlation calculation method (currently only correlation supported)')

# Data augmentation parameters
parser.add_argument('--gen_num', type=int, default=0)

# Save and load
parser.add_argument('--save_interval', type=int, default=50)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--vae_model_path', type=str, default=None)

# Inference parameters - consistent with DIFFT
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--max_seq_len', type=int, default=512, help='Deprecated: sequence length in feature selection task is determined by dataset feature count, this parameter is not used')
parser.add_argument('--infer_size', type=int, default=300)
parser.add_argument('--infer_func', type=str, default='reg')  # reg cls 
parser.add_argument('--infer_batch_size', type=int, default=64, help='Batch size during inference, can be set larger to speed up inference') 

args = parser.parse_args()


def compute_snr(noise_scheduler, timesteps):
    """
    Calculate Signal-to-Noise Ratio (SNR) - consistent with DIFFT
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr


def pre_training(ldm, vae, training_data, validation_data, infer_data, args):
    """Train Diffusion Model - consistent with DIFFT"""
    device = int(args.gpu)
    ldm.train()
    vae.eval()
    
    criterion = nn.MSELoss()
    start_epoch = 0
    best_val = 9999
    best_acc = 0
    val_loss = 9999
    infer_acc = 0
    
    optimizer = torch.optim.Adam(ldm.parameters(), lr=args.diff_lr)
    for group in optimizer.param_groups:
        group["initial_lr"] = args.diff_lr
    
    # Calculate total training steps: num_epochs * num_batches per epoch
    total_steps = args.diff_epochs * len(training_data)
    warmup_steps = total_steps // 20  # 5% of steps for warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps, 
        last_epoch=-1  # Start from scratch
    )
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        num_train_timesteps=args.diff_timesteps,
        prediction_type=args.prediction_type,
        thresholding=False,
    )
    
    # Save path
    model_path = args.model_path
    task_path = args.task_path
    
    # Gradient accumulation
    optimizer.zero_grad()
    print_log(f"Starting training from epoch {start_epoch} to {args.diff_epochs}, dataset size: {len(training_data)}", task_path)

    # Initialize loss record
    loss_history = []
    
    for epoch in range(start_epoch, args.diff_epochs):
        cost_time = 0
        train_loss = 0
        optimizer.zero_grad()
        
        print_log(f"Starting epoch {epoch}/{args.diff_epochs}...", task_path)
        
        for i, batch in enumerate(training_data):
            # Print progress every 20 batches
            if i % 20 == 0:
                print_log(f"  Epoch {epoch}, Batch {i}/{len(training_data)}", task_path)
            seq = batch["seqs"].to(device)
            tab = batch["tabs"].to(device)
            performance = batch["performances"].to(device)
            chunk = batch["chunk_seqs"]
            
            with torch.no_grad():
                # Use VAE encoding - consistent with DIFFT, supports tabular data
                encode_result = vae.encode(seq, tab)
                if len(encode_result) == 5:  # with tab_emb
                    z0, mu, logvar, predict_value, tab_emb = encode_result
                else:  # without tab_emb
                    z0, mu, logvar, predict_value = encode_result
                    tab_emb = tab  # fallback to original tab
                
                # Check the numerical range of z0 to avoid instability during training caused by excessively large values
                z0 = torch.clamp(z0, min=-10.0, max=10.0)
            
            start_time = time.time()
            # z0: seqlen,B,C -> B,seqlen,C
            if z0.dim() == 3 and z0.shape[0] != seq.shape[0]:
                z0 = z0.permute(1, 0, 2).contiguous()
            elif z0.dim() == 2:
                z0 = z0.unsqueeze(1)
            
            # Condition dropout - perfectly consistent with DIFFT (operates on original tab)
            # if random.random() < 0.1 and args.guidance_scale > 0:
            #     tab = torch.zeros_like(tab).float().to(device)
            
            cond = None  # Remove condition
            
            # Add noise
            noise = torch.randn_like(z0).to(device)
            bs = z0.shape[0]
            
            # Timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bs,),
                device=device,
            ).long()
            
            noisy_z = noise_scheduler.add_noise(z0, noise, timesteps)
            noise_pred = ldm(noisy_z, timesteps, cond=cond)
            
            # Get target
            if args.prediction_type == "epsilon":
                target = noise 
            elif args.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(z0, noise, timesteps)
            elif args.prediction_type == "sample":
                target = z0
            else:
                raise ValueError(f"Prediction Type: {args.prediction_type} not supported.")
            
            # Calculate loss
            if args.snr_gamma == 0:
                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, target, reduction="mean")
                elif args.loss_type in ["mse", "l2"]:
                    loss = F.mse_loss(noise_pred, target, reduction="mean")
                else:
                    raise ValueError(f"Loss Type: {args.loss_type} not supported.")
            else:
                # SNR weighted loss
                snr = compute_snr(noise_scheduler, timesteps)
                if args.loss_type == "l1":
                    loss = F.l1_loss(noise_pred, target, reduction="none")
                elif args.loss_type in ["mse", "l2"]:
                    loss = F.mse_loss(noise_pred, target, reduction="none")
                else:
                    raise ValueError(f"Loss Type: {args.loss_type} not supported.")
                
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                # Calculate weight based on prediction_type - fully consistent with DIFFT
                if args.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif args.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
            
            end_time = time.time()
            cost_time += (end_time - start_time)
            
            # Gradient accumulation - consistent with DIFFT (no extra loss scaling)
            loss.backward()
            
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(training_data):
                optimizer.step()
                scheduler.step()  # Call scheduler by steps, not by epoch
                optimizer.zero_grad()
            
            train_loss += loss.item()  # No need to multiply by accumulation_steps, it would scale up the loss value
        
        cost_time = time.time() - start_time
        # More frequent log output - every epoch
        current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.diff_lr
        print_log(f"[Epoch {epoch}] Loss: {train_loss/len(training_data):.6f}, lr: {current_lr:.8f}, time: {cost_time:.2f}s", task_path)

        # Record loss history
        epoch_loss = train_loss / len(training_data)
        loss_history.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })
        
        # Regular validation and saving
        if epoch % 10 == 0 :
            val_loss = validate_diffusion(ldm, vae, validation_data, device, args, epoch)
            # torch.save(ldm.state_dict(), os.path.join(model_path, f'ldm_{epoch}.pt'))
            
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ldm.state_dict(), os.path.join(model_path, f'ldm_best_val.pt'))
                print_log(f"Training Epoch {epoch} get best val_loss {best_val}, saving {os.path.join(model_path, f'ldm_best_val.pt')}", task_path)
        
        # Save latest model every epoch  
        torch.save(ldm.state_dict(), os.path.join(model_path, f'ldm_last.pt'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': ldm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': train_loss/len(training_data),
            }
            torch.save(checkpoint, os.path.join(model_path, f'checkpoint_last.pth'))

    # Save loss history
    loss_history_path = os.path.join(model_path, 'loss_history.json')
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print_log(f"Loss history saved to {loss_history_path}", task_path)

    # Plot loss curve
    try:
        plot_loss_curve(loss_history_path, model_path)
    except Exception as e:
        print_log(f"Failed to plot loss curve: {e}", task_path)

    # Save final model
    torch.save(ldm.state_dict(), os.path.join(model_path, f'ldm_final.pt'))
    return ldm


def validate_diffusion(ldm, vae, validation_data, device, args, epoch=0):
    """Validate Diffusion Model - refer to DIFFT implementation"""
    print_log(f'Validation Epoch [{epoch}] Start', args.task_path)
    
    # Initialize DiffTipeline - consistent with DIFFT
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        clip_sample=False, 
        beta_start=0.00085, 
        beta_end=0.012,
        steps_offset=1, 
        rescale_betas_zero_snr=True, 
        beta_schedule="scaled_linear"
    )
    
    pipeline = DiffTipeline(vae, scheduler, target_guidance=args.use_reward)
    generator = torch.Generator(device="cuda").manual_seed(0)
    
    vae.eval()
    ldm.eval()
    criterion = nn.MSELoss()
    loss = 0

    with torch.no_grad():
        for i, batch in enumerate(validation_data):
            seq = batch["seqs"].to(device)
            tab = batch["tabs"].to(device)
            performance = batch["performances"].to(device)
            chunk = batch["chunk_seqs"]
            
            # Use VAE encoding - consistent with DIFFT, supports tabular data
            encode_result = vae.encode(seq, tab)
            if len(encode_result) == 5:  # with tab_emb
                z0, mu, logvar, predict_value, tab_emb = encode_result
            else:  # without tab_emb
                z0, mu, logvar, predict_value = encode_result
                tab_emb = tab  # fallback to original tab
            
            # z0: seqlen,B,C -> B,seqlen,C (handles z0 with different dimensions)
            if z0.dim() == 3 and z0.shape[0] != seq.shape[0]:
                z0 = z0.permute(1, 0, 2).contiguous()
            elif z0.dim() == 2:
                z0 = z0.unsqueeze(1)
            
            # Condition dropout - perfectly consistent with DIFFT (operates on original tab)
            if random.random() < 0.1 and args.guidance_scale > 0:
                tab = torch.zeros_like(tab).float().to(device)
            
            cond = tab.unsqueeze(1)
            
            # Use Pipeline to generate latent vectors
            z_predict, z_list, _ = pipeline(
                ldm, z0.shape, cond, 
                steps=args.diff_num_step,  # consistent with DIFFT
                generator=generator, 
                guidance_scale=args.guidance_scale, 
                device=device, 
                use_reward=args.use_reward
            )

            loss += criterion(z_predict.float(), z0.float()).item()
    
    loss = loss / len(validation_data)  # avg validation loss by num_batches, not by num_samples
    print_log(f'Validation Epoch [{epoch}] Loss: {loss:.4f}', args.task_path)
    return loss


def infer(fe, vae, ldm, data, device, args):
    """Inference function - refer to DIFFT implementation"""
    data, df = data
    # Get real feature count - sequence length for feature selection task should equal feature count
    actual_feature_count = df.shape[1] - 1  # subtract label column
    history_csv_path = os.path.join(args.task_path, f'inference_history_{args.infer_func}.csv')
    # Evaluate initial performance on test set using fe
    initial_mask = np.ones(actual_feature_count, dtype=int)
    initial_test_data = fe.generate_data(initial_mask, 'test')
    max_acc = fe.get_performance(initial_test_data)
    print_log(f'Infer Start, Original accuracy on test set:{max_acc} (num_features: {actual_feature_count})', args.task_path)
    
    # y = df.iloc[:, -1] # no longer needed
    
    # Initialize DiffTipeline - consistent with DIFFT
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        clip_sample=False, 
        beta_start=0.00085, 
        beta_end=0.012,
        steps_offset=1, 
        rescale_betas_zero_snr=True, 
        beta_schedule="scaled_linear"
    )
    
    pipeline = DiffTipeline(vae, scheduler, target_guidance=args.use_reward)
    generator = torch.Generator(device="cuda").manual_seed(0)
    
    # Feature selection task: seq_len = actual_feature_count, not max_seq_len
    seq_len = actual_feature_count
    # shape will be dynamically set in each batch, no need to pre-define
    
    # df = df.iloc[:, :-1] # no longer needed
    # df.columns = [str(i) for i in range(df.shape[1])] # no longer needed
    vae.eval()
    ldm.eval()
    total_time = 0
    
    processed_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data, desc="Inferring")):
            seq = batch["seqs"]
            tab = batch["tabs"]
            performance = batch["performances"]
            chunk = batch["chunk_seqs"] 
            seq = seq.to(device)
            tab = tab.to(device)
            performance = performance.to(device)
            sample_time = time.time()

            # Condition handling - do not use random dropout during inference for stability and performance
            # if random.random() < 0.1 and args.guidance_scale > 0:
            #     tab = torch.zeros_like(tab).float().to(device)
            cond = None  # Remove condition
            
            # Dynamically adjust shape to support different batch sizes
            current_batch_size = seq.shape[0]  # modification: use seq batch size
            shape = (current_batch_size, seq_len, args.latent_dim)
                
            z_predict, z_list, _ = pipeline(
                ldm, shape, cond, 
                steps=args.diff_num_step,  # consistent with DIFFT, use diff_num_step
                generator=generator, 
                guidance_scale=args.guidance_scale, 
                device=device, 
                use_reward=args.use_reward
            )
            
            generated_seq = vae.generate(z_predict.permute(1, 0, 2).contiguous())
            # print(generated_seq)
            sample_time = time.time() - sample_time
            total_time += sample_time
            
            # Process each sequence in the batch separately and find best performance
            batch_best_acc = 0
            batch_best_df = None
            batch_best_features = 0
            pd.DataFrame(columns=['step', 'accuracy', 'num_features']).to_csv(history_csv_path, index=False)
            for seq_idx, seq_item in enumerate(generated_seq):
                try:
                    # Now sequence is in 0/1 mask format, interpret directly as feature selection mask
                    mask = seq_item.cpu().numpy().astype(int)

                    # Ensure mask length is correct
                    if len(mask) != actual_feature_count:
                        continue
                    
                    # If no features selected, skip
                    if not mask.any():
                        continue
                    
                    # Use fe.generate_data to create test set DataFrame
                    current_df = fe.generate_data(mask, 'test')
                    
                    # If generated data is empty (e.g. all feature columns deleted), skip
                    if current_df.shape[1] <= 1: # only label column
                        continue

                    # Calculate current accuracy
                    current_acc = fe.get_performance(current_df)
                    
                    # Keep best result
                    if current_acc > batch_best_acc:
                        batch_best_acc = current_acc
                        batch_best_df = current_df
                        batch_best_features = int(mask.sum())
                        
                except Exception as e:
                    error(f"Error during sequence evaluation: {e}")
                    continue
            
            # Check if better result is found
            if batch_best_acc > max_acc:
                print_log(f'New accuracy on test set: {batch_best_acc} (num_features: {batch_best_features}, batch best seq)', args.task_path)
                if batch_best_df is not None:
                    batch_best_df.to_csv(os.path.join(args.task_path, f'infer_{args.infer_func}_{batch_best_features}.csv'), index=False)
                max_acc = batch_best_acc
            else:
                print_log(f"current accuracy on test set: {batch_best_acc}")
                
            current_step = locals().get('i', locals().get('step', -1)) 
            
            step_record = {
                'step': current_step,
                'accuracy': batch_best_acc,
                'num_features': batch_best_features,
            }
            
            # Use append mode 'a' to write a line, header=False prevents redundant header writing
            pd.DataFrame([step_record]).to_csv(history_csv_path, mode='a', header=False, index=False)
            
            processed_batches += 1
    
    total_samples = processed_batches * args.infer_batch_size
    avg_sample_time = total_time / total_samples if total_samples > 0 else 0
    print_log(f'Infer Finished, Total Sample Time: {total_time:.4f} seconds, Single Sample Time: {avg_sample_time:.4f} seconds, Processed {processed_batches} batches', args.task_path)
    return max_acc


def plot_loss_curve(loss_history_path, save_dir):
    """Plot loss curve of Diffusion model"""
    try:
        with open(loss_history_path, 'r') as f:
            loss_history = json.load(f)

        epochs = [item['epoch'] for item in loss_history]
        train_loss = [item['train_loss'] for item in loss_history]
        val_loss = [item['val_loss'] for item in loss_history]
        lr = [item['lr'] for item in loss_history]

        plt.figure(figsize=(15, 5))

        # Plot train loss and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot train loss
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(epochs, lr, 'g-', label='Learning Rate', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plot_path = f'{save_dir}/loss_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print_log(f"Loss curve saved to {plot_path}", save_dir)

    except Exception as e:
        print_log(f"Error plotting loss curve: {e}", save_dir)
        plt.close('all')  # ensures all figures are closed to prevent memory leaks

def print_log(msg, log_path):
    """Simple log printing function"""
    print(msg)
    # Can optionally write to file
    # with open(os.path.join(log_path, 'log.txt'), 'a') as f:
    #     f.write(msg + '\n')

def main():
    if not torch.cuda.is_available():
        print_log('No GPU found!', '.')
        sys.exit(1)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    device = int(args.gpu)
    
    # Set paths - ensure paths are correct
    # Find project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # parent directory
    data_base_path = os.path.join(project_root, 'data', 'dataset')
    
    args.task_path = os.path.join(data_base_path, 'history', f'{args.task_name}_ldm', f'{args.method_name}_diffusion')
    args.model_path = os.path.join(args.task_path, 'model')
    os.makedirs(args.task_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)
    
    # Print parameters
    for arg, value in vars(args).items():
        print_log(f"{arg}: {value}", args.task_path)
    
    # Load feature evaluator
    fe_path = os.path.join(data_base_path, 'history', args.task_name, 'fe.pkl')
    print_log(f'Loading feature evaluator from: {fe_path}', args.task_path)
    with open(fe_path, 'rb') as f:
        fe: FeatureEvaluator = pickle.load(f)
    
    # Check and set inference function type - set before data processing
    if args.task_name.startswith('openml_'):
        args.infer_func = 'reg'
        print_log(f'Set infer_func to "reg" for OpenML dataset: {args.task_name}', args.task_path)
    
    # Prepare dataset, now simultaneously generating tab data
    if hasattr(fe, 'get_record_with_tabs'):
        # Use new method to generate tab data
        choice, labels, tabs = fe.get_record_with_tabs(args.gen_num, eos=fe.ds_size)
        valid_choice, valid_labels, valid_tabs = fe.get_record_with_tabs(0, eos=fe.ds_size)
        print_log(f'Successfully generated tab data: train={len(tabs)}, valid={len(valid_tabs)}', args.task_path)
    else:
        # Fallback to original method
        choice, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
        valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)
        tabs, valid_tabs = None, None
        print_log('Warning: tab data generation not available, using fallback', args.task_path)
    
    print_log(f'Note: VAE model now expects 0/1 mask format, DiffusionDataset will convert sequence to mask automatically', args.task_path)
    
    # Normalize labels - consistent with train_controller.py processing
    print_log(f'Labels type: {type(labels)}, length: {len(labels)}', args.task_path)
    print_log(f'Valid labels type: {type(valid_labels)}, length: {len(valid_labels)}', args.task_path)
    
    # Simplified label processing - consistent with train_controller.py
    min_val = min(labels)
    max_val = max(labels)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]
    
    # Create adapter dataset - consistent with DIFFT format, supports real tab data
    class DiffusionDataset(torch.utils.data.Dataset):
        def __init__(self, choices, targets, tabs=None, fe=None, train=True, ds_size=None):
            self.choices = choices
            self.targets = targets
            self.tabs = tabs
            self.fe = fe
            self.train = train
            # Ensure ds_size is accessible
            self.ds_size = ds_size or (fe.ds_size if fe and hasattr(fe, 'ds_size') else 25)
            
        def __len__(self):
            return len(self.choices)
            
        def __getitem__(self, index):
            choice = self.choices[index]
            target = self.targets[index] if self.targets is not None else 0.0
            
            # Get tab data
            if self.tabs is not None:
                tab_data = self.tabs[index]
                if not isinstance(tab_data, torch.Tensor):
                    tab_data = torch.tensor(tab_data, dtype=torch.float32)
            else:
                # Fallback solution: generate tab data based on choice
                if self.fe is not None and hasattr(self.fe, 'generate_tab_data'):
                    # Convert choice to binary mask and generate tab
                    if isinstance(choice, torch.Tensor):
                        choice_np = choice.numpy()
                    else:
                        choice_np = np.array(choice)
                    
                    # Ensure conversion to binary mask format (since VAE now expects 0/1 mask)
                    if len(choice_np) != self.fe.ds_size:
                        # input is index sequence, needs conversion to binary mask
                        binary_mask = np.zeros(self.fe.ds_size, dtype=np.int64)
                        for c in choice_np:
                            if isinstance(c, (torch.Tensor, np.ndarray)):
                                c = c.item()
                            if 0 <= c < self.fe.ds_size:
                                binary_mask[c] = 1
                        choice_np = binary_mask
                    else:
                        # if length already matches, ensure it is in 0/1 format
                        choice_np = np.array(choice_np, dtype=np.int64)
                     
                    tab_data = torch.tensor(self.fe.generate_tab_data(choice_np), dtype=torch.float32)
                else:
                    # Final fallback: use original feature count as condition dimension
                    if self.fe is not None and hasattr(self.fe, 'ds_size'):
                        tab_data = torch.randn(self.fe.ds_size)
                    else:
                        # Use ds_size property of instance
                        tab_data = torch.randn(self.ds_size)
            
            # Ensure choice is converted to 0/1 mask format (since VAE now expects this format)
            if isinstance(choice, torch.Tensor):
                choice_np = choice.numpy()
            else:
                choice_np = np.array(choice)
            
            # Debug info: check format of input choice
            # if index < 3:  # Only print debug info for first 3 samples
            #     print(f"DEBUG: choice[{index}] original: {choice_np[:10]}... (len={len(choice_np)}, max={choice_np.max() if len(choice_np) > 0 else 'empty'}, min={choice_np.min() if len(choice_np) > 0 else 'empty'})")
            #     print(f"DEBUG: ds_size={self.ds_size}")
                 
            # Convert index sequence to binary mask
            if len(choice_np) != self.ds_size:
                # Input is index sequence, convert to binary mask
                binary_mask = np.zeros(self.ds_size, dtype=np.int64)
                for c in choice_np:
                    if isinstance(c, (torch.Tensor, np.ndarray)):
                        c = c.item()  
                    if 0 <= c < self.ds_size:
                        binary_mask[c] = 1
                    # elif index < 3:  # Debug info
                    #     print(f"DEBUG: Invalid index {c} (should be 0 <= c < {self.ds_size})")
                choice_np = binary_mask
            else:
                # If length already matches, check for out-of-range values
                max_val = choice_np.max() if len(choice_np) > 0 else -1
                min_val = choice_np.min() if len(choice_np) > 0 else -1
                if max_val >= 2 or min_val < 0:
                    # if index < 3:
                    #     print(f"DEBUG: choice[{index}] has invalid values for 0/1 mask: max={max_val}, min={min_val}")
                    # Clean all non-0/1 values to 0/1
                    choice_np = np.clip(choice_np, 0, 1).astype(np.int64)
                else:
                    choice_np = np.array(choice_np, dtype=np.int64)
             
            # Final validation: ensure all values are in range [0, 1]
            if choice_np.max() > 1 or choice_np.min() < 0:
                print(f"ERROR: choice[{index}] final values out of range: {choice_np}")
                choice_np = np.clip(choice_np, 0, 1).astype(np.int64)
                 
            seqs_tensor = torch.tensor(choice_np, dtype=torch.long)
            chunk_seqs_tensor = torch.tensor(choice_np, dtype=torch.long)
                 
            return {
                "seqs": seqs_tensor,
                "tabs": tab_data,
                "performances": torch.tensor([target], dtype=torch.float),
                "chunk_seqs": chunk_seqs_tensor  # simplified processing
            }
    
    # Create dataset, passing tab data and ds_size parameter
    train_dataset = DiffusionDataset(choice, train_encoder_target, tabs=tabs, fe=fe, train=True, ds_size=fe.ds_size)
    valid_dataset = DiffusionDataset(valid_choice, valid_encoder_target, tabs=valid_tabs, fe=fe, train=True, ds_size=fe.ds_size)
    
    # Prepare inference dataset
    if hasattr(fe, 'get_record_with_tabs'):
        infer_choice, infer_labels, infer_tabs = fe.get_record_with_tabs(args.infer_size, eos=fe.ds_size)
    else:
        infer_choice, infer_labels = fe.get_record(args.infer_size, eos=fe.ds_size)
        infer_tabs = None
    
    infer_encoder_target = [(i - min_val) / (max_val - min_val) for i in infer_labels]
    infer_dataset = DiffusionDataset(infer_choice, infer_encoder_target, tabs=infer_tabs, fe=fe, train=False)
    infer_loader = DataLoader(infer_dataset, batch_size=args.infer_batch_size, shuffle=False, pin_memory=True)
    
    # Prepare original DataFrame for inference
    df = fe.original  # FeatureEvaluator uses .original property to store original data
    infer_data = (infer_loader, df)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # Get dataset parameters - use DIFFT style Feature_GCN
    # tab_len = num_samples (consistent with DIFFT)
    args.tab_len = fe.ds_size  # use num_samples as tab_len (Feature_GCN output dimension)
    args.latent_dim = args.d_latent_dim  # add latent_dim parameter for inference function
    
    print_log(f'Dataset info: ds_size={fe.ds_size}, num_samples={fe.original.shape[0]}, tab_len={args.tab_len}', args.task_path)
    print_log(f'Model path: {args.model_path}', args.task_path)
    
    # Load pretrained VAE model
    vae = GAFS(fe, args)
    vae_path = args.vae_model_path or os.path.join(data_base_path, 'history', args.task_name, f'GAFS_{args.method_name}.model_dict')
    
    if os.path.exists(vae_path):
        # Load pretrained weights, skip TabEncoder parameters - consistent with DIFFT
        state_dict = torch.load(vae_path, map_location=torch.device("cuda"))
        
        # Filter out TabEncoder related parameters
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('tab_encoder.')}
        
        # Load filtered parameters
    missing_keys, unexpected_keys = vae.load_state_dict(filtered_state_dict, strict=False)
 
    print_log(f"Loaded VAE model from {vae_path}", args.task_path)
    print_log(f"TabEncoder parameters skipped (will use random initialization)", args.task_path)
 
    if missing_keys:
        # only report non-TabEncoder missing keys
        non_tab_missing = [k for k in missing_keys if not k.startswith('tab_encoder.')]
        if non_tab_missing:
            print_log(f"Other missing keys: {non_tab_missing[:3]}{'...' if len(non_tab_missing) > 3 else ''}", args.task_path)
    else:
        print_log(f"VAE model not found at {vae_path}", args.task_path)
        sys.exit(1)
    
    vae = vae.cuda(device)
    vae.eval()
    
    # Initialize structured sparse attention mask
    mask_generator = None
    if args.use_sparse_mask:
        print_log(f"🔧 Initializing structured sparse attention mask (top_k={args.sparse_top_k}, method={args.sparse_method})", args.task_path)
         
        # Generate mask using original data from FeatureEvaluator
        mask_generator = FeatureMaskGenerator(
            data=fe.original,
            top_k=args.sparse_top_k,
            method=args.sparse_method
        )
         
        # Set mask for VAE
        vae.set_sparse_mask(mask_generator)
        print_log(f"✅ VAE Encoder set with sparse mask (Decoder remains original)", args.task_path)
    
    # Initialize Diffusion Model - use TransformerDM to stay consistent with DIFFT
    print_log(f"VAE latent dim: {args.d_latent_dim}, tab_len: {args.tab_len}", args.task_path)
    ldm = TransformerDM(
        in_channels=args.d_latent_dim, 
        t_channels=256, 
        context_channels=args.diff_hidden_size,  # cond_tab_encoder output dimension
        hidden_channels=args.diff_hidden_size, 
        depth=args.diff_num_layers,
        dropout=args.diff_dropout, 
        tab_len=args.tab_len,  # use original feature count
        out_channels=None
    ).to(device)
    
    print_log(f"Diffusion model parameters: {sum(p.numel() for p in ldm.parameters()) / 1e6:.2f}M", args.task_path)
    
    # Set sparse mask for Diffusion Model
    if args.use_sparse_mask and mask_generator is not None:
        # Diffusion Model uses additive mask (-inf indicates masked)
        diffusion_mask = mask_generator.get_diffusion_mask(device=device)
        ldm.set_sparse_mask(diffusion_mask)
        print_log(f"✅ Diffusion Model set with sparse mask", args.task_path)
    
    if args.test:
        print_log("Start Infering", args.task_path)
        # For test mode, need to load pretrained LDM model
        ldm_path = os.path.join(data_base_path, 'history', f'{args.task_name}_ldm', f'{args.method_name}_diffusion', 'model', 'ldm_best_val.pt')
        if os.path.exists(ldm_path):
            try:
                ldm.load_state_dict(torch.load(ldm_path, map_location=torch.device("cuda")))
                print_log(f"Load ldm model (strict) from {ldm_path}", args.task_path)
            except RuntimeError as e:
                # if strict load fails, try non-strict load
                state_dict = torch.load(ldm_path, map_location=torch.device("cuda"))
                missing_keys, unexpected_keys = ldm.load_state_dict(state_dict, strict=False)
                print_log(f"Load ldm model (non-strict) from {ldm_path}", args.task_path)
                if missing_keys:
                    print_log(f"LDM missing keys: {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}", args.task_path)
        else:
            print_log(f"LDM model not found at {ldm_path}", args.task_path)
            sys.exit(1)
        
        torch.save(ldm.state_dict(), os.path.join(args.model_path, f'ldm.pt'))
        torch.save(vae.state_dict(), os.path.join(args.model_path, f'vae.pt'))
        infer(fe, vae, ldm, infer_data, device, args)
        print_log("Infering Finished", args.task_path)
    else:
        # Start training
        print_log("Start pre-training", args.task_path)
        ldm = pre_training(ldm, vae, train_loader, valid_loader, infer_data, args=args)


if __name__ == '__main__':
    main()
