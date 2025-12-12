import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from model.EDSR import edsr_r16f64, edsr_r32f256
from utils.dataset import WebcamSRDataset
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

TEST_IMAGES_DIR = config['data']['test']['images_dir']
TEST_BATCH_SIZE = config['data']['test']['batch_size']

MODEL_SIZE = config["training"]["model_size"]
SCALE = config["training"]["scale"]
EPOCHS = config['training']['epochs']
DEVICE = config['training']['device']

MODEL_CHECKPOINT = config['model']['model_checkpoint']

if DEVICE == "cuda:0":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

test_data = WebcamSRDataset(root_dir=TEST_IMAGES_DIR, scale_factor=SCALE)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)


# Load model
if MODEL_SIZE == "large":
    model = edsr_r32f256(scale=SCALE)
else:
    model = edsr_r16f64(scale=SCALE)

model.load_pretrained(MODEL_CHECKPOINT)

model = model.to(DEVICE)

model.eval()

total_psnr = 0
total_ssim = 0

with torch.no_grad():
    for lr_data, hr_data in test_loader:
        lr_data, hr_data = lr_data.to(DEVICE), hr_data.to(DEVICE)

        output = model(lr_data)

