import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import itertools
import time
from DRCT import DRCT
from dataset import CarImageDataset

BATCH_SIZE = 32
UPSCALE_FACTOR = 2
HR_SIZE = 128

def evaluate_model(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for lr_img, hr_img in dataloader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr_img = model(lr_img)
            mse = nn.MSELoss()(sr_img, hr_img).item()
            mae = nn.L1Loss()(sr_img, hr_img).item()
            total_mse += mse
            total_mae += mae
    avg_mse = total_mse / len(dataloader)
    avg_mae = total_mae / len(dataloader)
    model.train()
    return avg_mse, avg_mae


def train_DRCT(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    total_iterations_phase1: int = 800000,
    total_iterations_phase2: int = 800000,
    initial_lr: float = 2e-4,
    milestones_phase1: list = [300000, 500000, 650000, 700000, 750000],
    milestones_phase2: list = [300000, 500000, 650000, 700000, 750000],
    log_interval: int = 1000,
    eval_interval: int = 5000,
    save_path_phase1: str = "drct_phase1.pth",
    save_path_phase2: str = "drct_phase2.pth",
    max_time: int = torch.inf
):
    start_time = time.time()
    timeout = False

    model.to(device)
    model.train()

    # Define loss functions
    criterion_L1 = nn.L1Loss().to(device)
    criterion_L2 = nn.MSELoss().to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))

    # Define learning rate schedulers for both phases
    scheduler_phase1 = MultiStepLR(optimizer, milestones=milestones_phase1, gamma=0.5)
    scheduler_phase2 = MultiStepLR(optimizer, milestones=milestones_phase2, gamma=0.5)

    # Create an infinite iterator for the dataloader
    train_data_iter = itertools.cycle(train_dataloader)

    # Training Phase 1: Fine-Tuning with L1 Loss
    print("Starting Phase 1: Fine-Tuning with L1 Loss")
    total_loss_phase1 = 0.0
    log_counter_phase1 = 0
    for iter in range(1, total_iterations_phase1 + 1):
        lr_img, hr_img = next(train_data_iter)
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        optimizer.zero_grad()
        sr_img = model(lr_img)
        loss = criterion_L1(sr_img, hr_img)
        loss.backward()
        optimizer.step()
        scheduler_phase1.step()

        total_loss_phase1 += loss.item()
        log_counter_phase1 += 1

        if iter % log_interval == 0:
            average_loss = total_loss_phase1 / log_counter_phase1
            print(f"Phase 1 Iteration {iter}/{total_iterations_phase1}, Average Loss: {average_loss:.6f}")
            total_loss_phase1 = 0.0
            log_counter_phase1 = 0

        if iter % eval_interval == 0:
            avg_mse, avg_mae = evaluate_model(model, val_dataloader, device)
            print(f"Phase 1 Iteration {iter}/{total_iterations_phase1}, Validation MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")

        # Check if max_time has been exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            print("Max time reached. Stopping training.")
            timeout = True
            break

    # Save model after Phase 1
    torch.save(model.state_dict(), save_path_phase1)
    print(f"Phase 1 complete. Model saved to {save_path_phase1}")
    
    if timeout:
        return

    # Training Phase 2: Refinement with L2 Loss
    print("Starting Phase 2: Refinement with L2 Loss")
    total_loss_phase2 = 0.0
    log_counter_phase2 = 0
    for iter in range(1, total_iterations_phase2 + 1):
        lr_img, hr_img = next(train_data_iter)
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        optimizer.zero_grad()
        sr_img = model(lr_img)
        loss = criterion_L2(sr_img, hr_img)
        loss.backward()
        optimizer.step()
        scheduler_phase2.step()

        total_loss_phase2 += loss.item()
        log_counter_phase2 += 1

        if iter % log_interval == 0:
            average_loss = total_loss_phase2 / log_counter_phase2
            print(f"Phase 2 Iteration {iter}/{total_iterations_phase2}, Average MSE: {average_loss:.6f}")
            total_loss_phase2 = 0.0
            log_counter_phase2 = 0

        if iter % eval_interval == 0:
            avg_mse, avg_mae = evaluate_model(model, val_dataloader, device)
            print(f"Phase 2 Iteration {iter}/{total_iterations_phase2}, Validation MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")

        # Check if max_time has been exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            print("Max time reached. Stopping training.")
            break

    # Save model after Phase 2
    torch.save(model.state_dict(), save_path_phase2)
    print(f"Phase 2 complete. Model saved to {save_path_phase2}")

    print("Training complete.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Constraints: img_size % (window_size * patch_size) == 0, embed_dim % num_heads == 0
    # Not necessary, but it's better to be: gc % num_heads == 0
    model = DRCT(
        input_size=HR_SIZE // UPSCALE_FACTOR,
        patch_size=4,
        in_chans=3,
        embed_dim=60,
        depths=[4],
        num_heads=10,
        window_size=4,
        mlp_ratio=3.125492,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.003572,
        attn_drop_rate=0.122857,
        drop_path_rate=0.093707,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        upscale=UPSCALE_FACTOR,
        img_range=1.,
        gc=30
    )
    model.to(device)

    train_dataset = CarImageDataset(root_dir="cropped_images", split="train", upscale_factor=UPSCALE_FACTOR)
    val_dataset = CarImageDataset(root_dir="cropped_images", split="val", upscale_factor=UPSCALE_FACTOR)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    train_DRCT(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        total_iterations_phase1=25000,
        total_iterations_phase2=5000,
        initial_lr=0.000476,
        milestones_phase1=[15000, 20000, 24000],
        milestones_phase2=[1500, 2000, 2500],
        log_interval=200,
        eval_interval = 5000,
        save_path_phase1="drct_phase1.pth",
        save_path_phase2="drct_phase2.pth"
    )
