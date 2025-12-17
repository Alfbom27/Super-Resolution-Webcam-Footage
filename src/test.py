import torch
from torch.utils.data import DataLoader
import argparse
import yaml
from model.EDSR import edsr_r16f64, edsr_r32f256, edsr_r16f64_irb, edsr_r32f256_irb
from utils.dataset import WebcamSRDataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity


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
USE_INVERSE_RESIDUAL = config['model'].get('use_inverse_residual', False)
EXPANSION_FACTOR = config['model'].get('expansion_factor', 6)
IRB_REPEATS = config['model'].get('irb_repeats', 1)
IRB_VERSION = config['model'].get('irb_version', 'v2')
SQUEEZE_EXCITATION = config['model'].get('squeeze_excitation', True)

if DEVICE == "cuda:0":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

test_data = WebcamSRDataset(root_dir=TEST_IMAGES_DIR, scale_factor=SCALE, patch_size=None)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)


# Load model
if USE_INVERSE_RESIDUAL:
    if MODEL_SIZE == "large":
        model = edsr_r32f256_irb(scale=SCALE, expansion_factor=EXPANSION_FACTOR,
                                  irb_repeats=IRB_REPEATS, irb_version=IRB_VERSION,
                                  squeeze_excitation=SQUEEZE_EXCITATION)
    else:
        model = edsr_r16f64_irb(scale=SCALE, expansion_factor=EXPANSION_FACTOR,
                                 irb_repeats=IRB_REPEATS, irb_version=IRB_VERSION,
                                 squeeze_excitation=SQUEEZE_EXCITATION)
else:
    if MODEL_SIZE == "large":
        model = edsr_r32f256(scale=SCALE)
    else:
        model = edsr_r16f64(scale=SCALE)

model.load_pretrained(MODEL_CHECKPOINT)

model = model.to(DEVICE)

model.eval()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(DEVICE)

with torch.no_grad():
    for lr_data, hr_data in test_loader:
        lr_data, hr_data = lr_data.to(DEVICE), hr_data.to(DEVICE)

        output = model(lr_data)

        # Clamp output to [0, 1] range for metrics
        output = torch.clamp(output, 0.0, 1.0)

        psnr.update(output, hr_data)
        ssim.update(output, hr_data)
        lpips.update(output, hr_data)

    test_psnr = psnr.compute().item()
    test_ssim = ssim.compute().item()
    test_lpips = lpips.compute().item()

    print(
        f"Test PSNR: {test_psnr:.4f} | "
        f"Test SSIM: {test_ssim:.4f} | "
        f"Test LPIPS: {test_lpips:.4f} | "
    )
