import numpy as np
from PIL import Image
import io


class WebcamDegradation:

    def __init__(self, downscale_factor=2, noise_sigma=5.0, jpeg_quality=75,
                 gamma=1.5, gains=None):
        self.downscale_factor = downscale_factor
        self.noise_sigma = noise_sigma
        self.jpeg_quality = int(np.clip(jpeg_quality, 5, 95))
        self.gamma = gamma
        self.gains = np.array(gains if gains is not None else [1.0, 1.0, 1.0])

    def downscale(self, img_array):
        if self.downscale_factor > 1:
            h, w = img_array.shape[:2]
            new_w = max(32, w // self.downscale_factor)
            new_h = max(32, h // self.downscale_factor)
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            img_pil = img_pil.resize((new_w, new_h), Image.BICUBIC)
            return np.array(img_pil).astype(np.float32) / 255.0
        return img_array

    def apply_white_balance(self, img):
        balanced = img * self.gains.reshape(1, 1, 3)
        #if balanced.max() > 1.0:
        #    balanced /= balanced.max()
        return np.clip(balanced, 0.0, 1.0)

    def add_noise(self, img):
        noise = np.random.normal(0, self.noise_sigma / 255.0, img.shape)
        return np.clip(img + noise, 0.0, 1.0)

    def apply_gamma(self, img):
        return np.power(img, 1.0 / max(self.gamma, 0.1))

    def apply_jpeg_compression(self, img):
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=self.jpeg_quality)
        buffer.seek(0)
        img_compressed = Image.open(buffer)
        return np.array(img_compressed).astype(np.float32) / 255.0

    def __call__(self, img_pil):
        """Apply full degradation pipeline to PIL Image."""
        # Convert PIL to numpy array (RGB)
        img = np.array(img_pil).astype(np.float32) / 255.0

        img = self.downscale(img)
        img = self.apply_white_balance(img)
        img = self.add_noise(img)
        img = self.apply_gamma(img)
        # img = self.apply_jpeg_compression(img)

        # Convert back to uint8 for PIL
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(img_uint8)
