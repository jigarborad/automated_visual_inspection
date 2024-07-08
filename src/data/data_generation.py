import numpy as np
import cv2
from typing import Tuple, List

def generate_synthetic_image(size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Generate a synthetic image of a product."""
    image = np.ones((*size, 3), dtype=np.uint8) * 255
    # Add some random shapes to simulate a product
    cv2.rectangle(image, (50, 50), (150, 150), (200, 200, 200), -1)
    cv2.circle(image, (100, 100), 30, (180, 180, 180), -1)
    return image

def add_defect(image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Add a random defect to the image and return the defect location."""
    defect_x = np.random.randint(0, image.shape[1])
    defect_y = np.random.randint(0, image.shape[0])
    cv2.circle(image, (defect_x, defect_y), 5, (0, 0, 255), -1)
    return image, [defect_x, defect_y]

def generate_dataset(num_images: int) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Generate a dataset of images with defects."""
    images = []
    labels = []
    for _ in range(num_images):
        image = generate_synthetic_image()
        image, defect_location = add_defect(image)
        images.append(image)
        labels.append(defect_location)
    return images, labels