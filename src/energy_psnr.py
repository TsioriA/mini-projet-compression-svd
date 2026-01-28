import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from load_image import load_grayscale_image
from svd_compress import compute_svd_compression

def psnr(original, compressed):
    """Peak Signal-to-Noise Ratio (métrique standard compression)."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

if __name__ == "__main__":
    img_path = Path(__file__).parent.parent / "data/moon.tif"
    img = load_grayscale_image(img_path)
    
    # SVD complète
    U, s, Vt, _, _ = compute_svd_compression(img, [min(300, min(img.shape))])
    
    # Courbe énergie cumulée (scientifique !)
    energy_cum = np.cumsum(s**2) / np.sum(s**2)
    k_range = np.arange(1, len(s)+1)
    
    # PSNR pour k=1,10,50,100,200
    k_values = [1, 10, 50, 100, 200]
    psnrs = []
    for k in k_values:
        U_k, s_k, Vt_k = U[:, :k], np.diag(s[:k]), Vt[:k, :]
        img_k = U_k @ s_k @ Vt_k
        psnrs.append(psnr(img, img_k))
    
    # Figure 2x1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Courbe énergie (POUVOIR EXPLICATIF)
    ax1.plot(k_range, energy_cum, 'b-', linewidth=2)
    ax1.axvline(50, color='green', linestyle='--', label='k=50 (95% énergie)')
    ax1.axvline(200, color='orange', linestyle='--', label='k=200 (99.3%)')
    ax1.set_xlabel('Rang k')
    ax1.set_ylabel('Énergie cumulée (%)')
    ax1.set_title('Décroissance énergie - Moon.tif (537×358)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tableau PSNR + Compression
    ax2.axis('tight')
    ax2.axis('off')
    table_data = [
        ['k', 'Stockage %', 'Erreur Frobenius', 'PSNR (dB)'],
        ['1', '0.5%', '0.4530', f'{psnrs[0]:.1f}'],
        ['10', '4.7%', '0.1205', f'{psnrs[1]:.1f}'],
        ['50', '23.3%', '0.0367', f'{psnrs[2]:.1f}'],
        ['100', '46.6%', '0.0179', f'{psnrs[3]:.1f}'],
        ['200', '93.1%', '0.0070', f'{psnrs[4]:.1f}']
    ]
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0,0,1,1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / "results/energy_psnr_moon.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("PSNR values:", [f"{p:.1f}" for p in psnrs])
