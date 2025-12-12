import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from model.EDSR import edsr_r16f64, edsr_r32f256
from utils.dataset import WebcamSRDataset
from utils.loss import L1Loss, PerceptualLoss, CombinedLoss
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to YAML config file",
)

args = parser.parse_args()

with open(args.config, 'r') as file:
    config=yaml.safe_load(file)

TRAIN_IMAGES_DIR = config['data']['train']['images_dir']
TRAIN_BATCH_SIZE = config['data']['train']['batch_size']

VALIDATION_IMAGES_DIR = config['data']['valid']['images_dir']
VALIDATION_BATCH_SIZE = config['data']['valid']['batch_size']

MODEL_SIZE = config["training"]["model_size"]
SCALE = config["training"]["scale"]
LEARNING_RATE = config['training']['optimizer']['lr']
EPOCHS = config['training']['epochs']
DEVICE = config['training']['device']
COMBINED_LOSS = config['training']['combined_loss']

MODEL_CHECKPOINT = config['model']['model_checkpoint']
TRAIN_FROM_CHECKPOINT = config['model']['pre_trained']

if DEVICE == "cuda:0":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

# Dataset
# Add transforms
train_data = WebcamSRDataset(root_dir=TRAIN_IMAGES_DIR, scale_factor=SCALE)
val_data = WebcamSRDataset(root_dir=VALIDATION_IMAGES_DIR, scale_factor=SCALE)

# Dataloader
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

# Load model
if MODEL_SIZE == "large":
    model = edsr_r32f256(scale=SCALE)
else:
    model = edsr_r16f64(scale=SCALE)

if TRAIN_FROM_CHECKPOINT:
    model.load_pretrained(MODEL_CHECKPOINT)

model = model.to(DEVICE)

# Optimizer and lr scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)

# loss

if COMBINED_LOSS:
    l1_loss = L1Loss()
    perceptual_loss = PerceptualLoss(shift=1)
    loss_fn = CombinedLoss(pixel_loss=l1_loss, perceptual_loss=perceptual_loss, pixel_weight=1.0, perceptual_weight=0.1)
else:
    loss_fn = L1Loss()

# training loop

patience = 10

total_loss = []
total_val_loss = []

best_val_loss = np.inf
epochs_without_improvement = 0

psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

total_psnr = []
total_ssim = []
total_val_psnr = []
total_val_ssim = []

for epoch in range(EPOCHS):
    model.train()

    psnr.reset()
    ssim.reset()

    train_loss = 0
    val_loss = 0

    for lr_data, hr_data in train_loader:
        lr_data, hr_data = lr_data.to(DEVICE), hr_data.to(DEVICE)


        optimizer.zero_grad()

        output = model(lr_data)

        loss = loss_fn(output, hr_data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        psnr.update(output, hr_data)
        ssim.update(output, hr_data)

    train_loss /= len(train_loader)
    total_loss.append(train_loss)

    train_psnr = psnr.compute().item()
    train_ssim = ssim.compute().item()

    total_psnr.append(train_psnr)
    total_ssim.append(train_ssim)

    model.eval()

    psnr.reset()
    ssim.reset()
    with torch.no_grad():
        for lr_data, hr_data in val_loader:
            lr_data, hr_data = lr_data.to(DEVICE), hr_data.to(DEVICE)

            output = model(lr_data)

            loss = loss_fn(output, hr_data)

            val_loss += loss.item()
            psnr.update(output, hr_data)
            ssim.update(output, hr_data)



    val_loss /= len(val_loader)
    total_val_loss.append(val_loss)

    val_psnr = psnr.compute().item()
    val_ssim = ssim.compute().item()

    total_val_psnr.append(val_psnr)
    total_val_ssim.append(val_ssim)

    lr_scheduler.step()

    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train PSNR: {train_psnr:.4f} | "
        f"Val PSNR: {val_psnr:.4f} | "
        f"Train SSIM: {train_ssim:.4f} | "
        f"Val SSIM: {val_ssim:.4f} | "
        f"Current LR: {optimizer.param_groups[0]['lr']}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break
