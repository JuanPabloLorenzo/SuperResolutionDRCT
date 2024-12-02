import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DRCT import DRCT
from training import CarImageDataset, evaluate_model, train_DRCT
import pickle

def objective(trial):
    # Constraints: img_size % (window_size * patch_size) == 0, embed_dim % num_heads == 0
    # Not necessary, but it's better to be: gc % num_heads == 0 and embed_dim % gc == 0
    depth = trial.suggest_int('depth', 4, 10)
    num_heads = trial.suggest_int('num_heads', 4, 12)
    embed_dim = trial.suggest_int('embed_dim', 40 - (40 % num_heads), 180 - (180 % num_heads), step=num_heads)
    window_size = trial.suggest_categorical('window_size', [4, 8])
    mlp_ratio = trial.suggest_float('mlp_ratio', 2.0, 5.0)
    drop_rate = trial.suggest_float('drop_rate', 0.0, 0.5)
    attn_drop_rate = trial.suggest_float('attn_drop_rate', 0.0, 0.4)
    drop_path_rate = trial.suggest_float('drop_path_rate', 0.0, 0.2)
    ape = trial.suggest_categorical('ape', [True, False])
    patch_norm = trial.suggest_categorical('patch_norm', [True, False])
    gc = trial.suggest_int('gc', 32 - (32 % num_heads), 60 - (60 % num_heads), step=num_heads)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    # Prepare dataset
    batch_size = 16
    patch_size = 4 #trial.suggest_categorical('patch_size', [2, 4])
    train_dataset = CarImageDataset(root_dir="cropped_images", split="train")
    val_dataset = CarImageDataset(root_dir="cropped_images", split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = DRCT(
        input_size=64,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depths=[depth],
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        ape=ape,
        patch_norm=patch_norm,
        upscale=2,
        img_range=1.,
        gc=gc,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define optimizer and other training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    inf = 1000000000
    train_DRCT(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        initial_lr=lr,
        milestones_phase1=[inf],
        milestones_phase2=[inf],
        log_interval=200,
        eval_interval=inf,
        device=device,
        total_iterations_phase1=inf,
        total_iterations_phase2=0,
        max_time=20*60,
    )
    
    # Evaluate the model (Assuming a validation loss is returned)
    mse, mae = evaluate_model(model, val_loader, device)  # Define this function as needed
    
    return mae

def save_study(study, trial):
    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)

import os
import pickle

if __name__ == "__main__":
    load_path = "study.pkl"
    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            study = pickle.load(f)
    else:
        study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, callbacks=[save_study])
    print("Best hyperparameters: ", study.best_params)
