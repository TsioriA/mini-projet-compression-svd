
# 🖼️ Compression d'image par SVD (Moon.tif 537×358)

**Mini-projet ML** - Janvier 2026  
**k=50 optimal : 23% stockage → 95% énergie → 96.3% qualité**

## 📊 Résultats clés

| k   | Stockage | Erreur Frobenius | Énergie cumulée |
|-----|----------|------------------|-----------------|
| **1**  | **0.5%** | 0.4530           | ~20%           |
| **10** | **4.7%** | 0.1205           | ~80%           |
| **50**| **23%**  | **0.0367**       | **95%** ✅      |
| 100  | 46.6%    | 0.0179           | 98%            |
| **200**| 93.1%  | **0.0070**       | **99.3%**      |

## 🚀 Installation & Test
```bash
pip install -r requirements.txt
python src/svd_compress.py     # Visualisations k=1..200
python src/energy_psnr.py      # Courbe énergie + PSNR
```

## 📈 Visualisations générées
- `results/svd_moon_k.png` : Reconstructions + cartes d'erreur
- `results/energy_psnr_moon.png` : **Courbe énergie (k=50=95%)**

## 👥 Membres du groupe
- Tsiori
- Tiantsoa
 

## 📚 Livrables
- **Rapport** : `report/rapport_final.pdf` 
- **Slides** : `slides/presentation.pdf`
- **Démo live** : 31/01/2026 7h-9h
```



