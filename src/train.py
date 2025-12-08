import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from model.EDSR import edsr_r16f64, edsr_r32f256
from utils.dataset import WebcamSRDataset
from utils.loss import L1Loss
import numpy as np

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

LEARNING_RATE = config['training']['optimizer']['lr']
EPOCHS = config['training']['epochs']
DEVICE = config['training']['device']

MODEL_CHECKPOINT = config['model']['model_checkpoint']

if DEVICE == "cuda:0":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

# Dataset
train_data = WebcamSRDataset(root_dir=TRAIN_IMAGES_DIR)
val_data = WebcamSRDataset(root_dir=VALIDATION_IMAGES_DIR)

# Dataloader
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)

# Load model
model = edsr_r16f64(scale=4)
model.load_pretrained(MODEL_CHECKPOINT)
model = model.to(DEVICE)

# Optimizer and lr scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)

# loss

loss_fn = L1Loss()

# training loop

patience = 10

total_loss = []
total_val_loss = []

best_val_loss = np.inf
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()

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

    train_loss /= len(train_loader)
    total_loss.append(train_loss)

    model.eval()
    with torch.no_grad():
        for lr_data, hr_data in val_loader:
            lr_data, hr_data = lr_data.to(DEVICE), hr_data.to(DEVICE)

            output = model(lr_data)

            loss = loss_fn(output, hr_data)
            val_loss += loss.item()


    val_loss /= len(val_loader)
    total_val_loss.append(val_loss)

    lr_scheduler.step()

    # Perhaps add logging psnr
    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
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
