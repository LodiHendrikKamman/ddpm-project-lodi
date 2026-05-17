# %% [markdown]
# # DDPM Example Usage
# 
# This script walks through the full workflow:
# setup → train → save/load → generate → plot.
# Make sure the `ddpm/` package folder is in the same directory (or on your PYTHONPATH).

# %% [markdown]
# cd "C:\Users\lodik\Documents\programming\diffusion-models-project-main"
# 
# git status
# git add ddpm example_usage_ddpm.ipynb changes.txt .gitignore
# git status
# git commit -m "Update DDPM project"
# git push

# %%
# --- Setup repo in Colab ---
%cd /content

import os
import sys

repo_path = "/content/ddpm-project-lodi"

if not os.path.exists(repo_path):
    !git clone https://github.com/LodiHendrikKamman/ddpm-project-lodi.git

%cd /content/ddpm-project-lodi
!git pull
!pip install -r requirements.txt

# Make sure Python can find the repo/package
if repo_path not in sys.path:
    sys.path.append(repo_path)

# %%
# --- Imports ---
import torch

from ddpm import NoiseScheduler, UNet, train, find_lr, generate_image, noisy_image
from ddpm.dataset import load_mnist, get_noisy_loaders
from ddpm.utils import load_unet, channel_list, model_name, path_name
from ddpm.viz import plot_generated
import random
import numpy as np

# --- Expected behaviouring ---
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Imports worked!")

# %% [markdown]
# ## 1. Noise scheduler and data
# 
# The scheduler defines the beta schedule and handles the forward noising process.
# T=1000 steps, linearly spaced betas from 1e-4 to 0.02 (Ho et al. defaults).

# %%
scheduler = NoiseScheduler(T=1000, beta_start=1e-4, beta_end=0.02).to(device)

train_set, test_set = load_mnist()
train_loader, test_loader = get_noisy_loaders(train_set, test_set, scheduler, batch_size=32)

# %% [markdown]
# ### Training on a single digit
# 
# Pass a filter function to get_noisy_loaders_filtered to restrict the dataset.
# Here we train on zeros only, but any (dataset -> Subset) function works.

# %%
from ddpm.dataset import get_noisy_loaders_filtered, zeros_only

train_loader_zeros, test_loader_zeros = get_noisy_loaders_filtered(
    train_set, test_set, scheduler, filter_fn=zeros_only, batch_size=32
)

unet_zeros = UNet(
    channels=channel_list(64),
    convs_per_level=2,
    num_heads_att=4,
    time_emb_dim=128,
    time_emb_base_dim=32,
).to(device)

train_losses, test_losses = train(
    unet_zeros,
    train_loader_zeros,
    test_loader_zeros,
    epochs=50,
    lr=1e-3,
    save_path='zeros_only_time.pkl',
    use_time=True
)

x = generate_image(unet_zeros, scheduler, n_images=8)
plot_generated(x, ncol=4)

# %% [markdown]
# ## 2. Build a UNet
# 
# `channels` sets the feature map depth at each encoder level.
# The decoder mirrors this automatically.
# `convs_per_level` is how many conv layers per resolution block. look at the documentation of UNet for a bit more detail (or ask Elias). The structure of the unet is basically taken from ronneberger et al., 2015 (the original unet paper, see also illustration in lilian wengs blog).
# 
# A reasonable small model to start with:

# %%
channels = channel_list(64)   # -> [64, 128, 256]
cpl = 2                       # convs per level

unet = UNet(channels=channels, convs_per_level=cpl).to(device)
print(f"Model: {model_name(64, cpl)}")
print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")

# %% [markdown]
# ## 3. Find a learning rate
# 
# Runs the LR range test and returns the suggested LR.
# Multiply by ~0.5 for a conservative starting point.

# %%
suggested_lr = find_lr(unet_zeros, train_loader_zeros, start_lr=1e-6, end_lr=1e-1, num_iter=100)
lr = suggested_lr * 0.5
print(f"Suggested LR: {suggested_lr:.2e} → using {lr:.2e}")
