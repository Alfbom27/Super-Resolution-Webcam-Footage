import cv2
import numpy as np


class WebcamDegradation:

    def __init__(self, downscale_factor=2, noise_sigma=5.0, jpeg_quality=75,
                 gamma=1.5, gains=None):
        self.downscale_factor = downscale_factor
        self.noise_sigma = noise_sigma
        self.jpeg_quality = int(np.clip(jpeg_quality, 25, 95))
        self.gamma = gamma
        self.gains = np.array(gains if gains is not None else [1.0, 1.0, 1.0])

    def downscale(self, img):
        if self.downscale_factor > 1:
            h, w = img.shape[:2]
            new_w = max(32, w // self.downscale_factor)
            new_h = max(32, h // self.downscale_factor)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def apply_white_balance(self, img):
        balanced = img * self.gains.reshape(1, 1, 3)
        if balanced.max() > 1.0:
            balanced /= balanced.max()
        return np.clip(balanced, 0.0, 1.0)

    def add_noise(self, img):
        noise = np.random.normal(0, self.noise_sigma / 255.0, img.shape)
        return np.clip(img + noise, 0.0, 1.0)

    def apply_gamma(self, img):
        return np.power(img, 1.0 / max(self.gamma, 0.1))

    def apply_jpeg_compression(self, img):
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        _, enc = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def __call__(self, img):
        """Apply full degradation pipeline to image (uint8 BGR)."""
        img = img.astype(np.float32) / 255.0

        img = self.downscale(img)
        img = self.apply_white_balance(img)
        img = self.add_noise(img)
        img = self.apply_gamma(img)
        img = self.apply_jpeg_compression(img)

        return img
