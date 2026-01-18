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

    def rgb_to_ycbcr_pil(self, img):
        return img.convert("YCbCr")

    def add_chroma_noise(self,
            img_np,
            sigma_y=2.0,
            sigma_c=6.0,
            blur_chroma=True
    ):
        img_pil = Image.fromarray(
            (img_np * 255.0).clip(0, 255).astype(np.uint8),
            mode="RGB"
        )
        if img_pil.mode != "YCbCr":
            img_pil = img_pil.convert("YCbCr")

        arr = np.asarray(img_pil).astype(np.float32)

        noise_y = np.random.randn(*arr[:, :, 0].shape) * sigma_y
        noise_cb = np.random.randn(*arr[:, :, 1].shape) * sigma_c
        noise_cr = np.random.randn(*arr[:, :, 2].shape) * sigma_c

        if blur_chroma:
            from scipy.ndimage import gaussian_filter
            noise_cb = gaussian_filter(noise_cb, sigma=1.0)
            noise_cr = gaussian_filter(noise_cr, sigma=1.0)

        arr[:, :, 0] += noise_y
        arr[:, :, 1] += noise_cb
        arr[:, :, 2] += noise_cr

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        img_out = Image.fromarray(arr, mode="YCbCr").convert("RGB")
        img_out = np.asarray(img_out).astype(np.float32) / 255.0

        return img_out

    def __call__(self, img_pil):
        """Apply full degradation pipeline to PIL Image."""
        img = np.array(img_pil).astype(np.float32) / 255.0

        img = self.add_noise(img)
        img = self.downscale(img)
        img = self.add_chroma_noise(
            img,
            sigma_y=1.0,
            sigma_c=5.0
        )
        img = self.apply_white_balance(img)
        self.noise_sigma = 4.0
        img = self.add_noise(img)
        img = self.apply_gamma(img)
        img = self.apply_jpeg_compression(img)

        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(img_uint8)
