import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from load_image import load_grayscale_image, display_image

def compute_svd_compression(img, k_values):
    """SVD + reconstructions rang k (numpy.linalg.svd AUTORISÉ)."""
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    
    reconstructions = {}
    errors = {}
    
    for k in k_values:
        U_k = U[:, :k]
        s_k = np.diag(s[:k])
        Vt_k = Vt[:k, :]
        img_k = U_k @ s_k @ Vt_k
        error = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
        reconstructions[k] = img_k
        errors[k] = error
    
    return U, s, Vt, reconstructions, errors

if __name__ == "__main__":
    img_path = Path(__file__).parent.parent / "data/moon.tif"
    img = load_grayscale_image(img_path)
    
    k_values = [1, 10, 50, 100, 200]
    U, s, Vt, reconstructions, errors = compute_svd_compression(img, k_values)
    
    print("Top 10 singular values:", s[:10])
    print("\nCompression:")
    for k in k_values:
        ratio = (k * (img.shape[0] + img.shape[1])) / (img.shape[0] * img.shape[1])
        print(f"k={k}: {ratio:.1%} ({errors[k]:.4f})")
    
    # Visualisation
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original (537x358)')
    axes[0,0].axis('off')
    
    for i, k in enumerate(k_values):
        axes[0,i+1].imshow(reconstructions[k], cmap='gray')
        axes[0,i+1].set_title(f'k={k}')
        axes[0,i+1].axis('off')
        axes[1,i+1].imshow(np.abs(img-reconstructions[k]), cmap='hot')
        axes[1,i+1].set_title(f'Error\n{errors[k]:.4f}')
        axes[1,i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / "results/svd_moon_k.png", dpi=150, bbox_inches='tight')
    plt.show()
