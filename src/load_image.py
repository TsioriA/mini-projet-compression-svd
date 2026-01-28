import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_grayscale_image(image_path):
    """Charge une image et la convertit en niveaux de gris (float 0-1)."""
    img = plt.imread(image_path)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)  # Conversion RGB vers gris
    img = img.astype(np.float64) / 255.0  # Normalisation [0,1]
    return img

def display_image(img, title="Image"):
    """Affiche l'image avec cmap gris."""
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Test
if __name__ == "__main__":
    # Votre image !
    img_path = Path(__file__).parent.parent / "data/moon.tif"
    img = load_grayscale_image(img_path)
    print(f"Shape: {img.shape}, Min: {img.min():.3f}, Max: {img.max():.3f}")
    display_image(img, "Moon - Image originale")
