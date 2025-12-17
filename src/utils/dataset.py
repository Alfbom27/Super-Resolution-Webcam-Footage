import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
from PIL import Image
from utils.transforms import WebcamDegradation

class WebcamSRDataset(Dataset):
    def __init__(self, root_dir, patch_size=48, scale_factor=4, noise_sigma=1.0, jpeg_quality=75, gamma=1.0):
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        if patch_size is not None:
            self.random_crop = T.RandomCrop(patch_size*scale_factor)

        self.degradation = WebcamDegradation(
            downscale_factor=scale_factor,
            noise_sigma=noise_sigma,
            jpeg_quality=jpeg_quality,
            gamma=gamma
        )

        self.image_paths = []
        for filename in os.listdir(root_dir):
            self.image_paths.append(os.path.join(root_dir, filename))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert('RGB')

        if self.patch_size is not None:
            hr_patch = self.random_crop(hr_img)
        else:
            hr_patch = hr_img

        lr_patch = self.degradation(hr_patch)

        hr_tensor = F.to_tensor(hr_patch)
        lr_tensor = F.to_tensor(lr_patch)

        return lr_tensor, hr_tensor
