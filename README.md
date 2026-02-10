# RGFS

## Step 0: Data Collection
Generate the training dataset for the vae, evaluator and diffusion model:
```bash
python baseline/automatic_feature_selection_gen.py
```

## Step 1: Train VAE+Evaluator
```bash
python ours/train_vae.py
```

## Step 2: Train Diffusion Model
Train the Transformer-based Latent Diffusion Model (LDM):
```bash
python ours/train_diffusion.py
```

## Final: Testing
To evaluate the trained model on the test set, use the `--test` flag:
```bash
python ours/train_diffusion.py --test
```
