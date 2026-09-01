# Détection de Route LiDAR AI36 - Architecture du Système

## 1. Vue d’Ensemble du Pipeline

```
/patchworkpp/ground
       │
       ▼
┌────────────────────────────────────────┐
│ [01] road_points_extraction.py         │
│  • Intensité ∈ [0.0, 8.0]              │
│  • Pipeline unifié basé sur DBSCAN     │
└────────────────────────────────────────┘
       │ /road_surface
       ▼
┌────────────────────────────────────────┐
│ [02] ring_boundary_extraction.py       │
│  • Anneaux 0-14 (max 15)               │
│  • Ajustement de courbe RANSAC (degré 2)│
│  • Contraintes géométriques + lissage  │
└────────────────────────────────────────┘
       │ /ring_boundaries
       ▼
┌────────────────────────────────────────┐
│ [03] rviz2_publisher.py                │
│  • Visualisation uniquement (LINE_STRIP)│
│  • Cyan (gauche) + Magenta (droite)    │
│  • MarkerArray (lignes de bordure)     │
└────────────────────────────────────────┘
       │ /road_boundaries
       ▼
     RViz2
```

**Nœuds supprimés :** curves_smoothing, marker_array, voxelisation  
**Fichiers réservés :** `03_roundabout_handler.py` (non lancé)

---

## 2. Détails des Nœuds

### 2.1 Patchwork++ (C++ Externe)
| Propriété | Valeur |
|-----------|--------|
| **Package** | `patchworkpp` |
| **Entrée** | `/velodyne_points` (PointCloud2) |
| **Sortie** | `/patchworkpp/ground` (PointCloud2) |
| **But** | Segmentation du plan de sol via GPF zonal |

### 2.2 Nœud 01 : Extraction des Points de Route ("Less is More")
| Propriété | Valeur |
|-----------|--------|
| **Fichier** | `01_road_points_extraction.py` (284 lignes) |
| **Entrée** | `/patchworkpp/ground` |
| **Sortie** | `/road_surface` (PointCloud2) |
| **Algorithme** | Pipeline unifié en 4 étapes basé sur DBSCAN |

**Étapes de l’algorithme :**
1. **Filtre d’intensité :** [0.0, 8.0] pour l’asphalte
2. **Filtre de densité :** validation des voisins via cKDTree
3. **Confirmation temporelle :** buffer de 5 images avec binning spatial
4. **Clustering DBSCAN :** suppression finale du bruit

**Paramètres clés :**
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `min_road_intensity` | 0.0 | Limite inférieure asphalte |
| `max_road_intensity` | 6.0 | Limite supérieure asphalte |
| `max_range` | 25.0 | Distance max du véhicule (m) |
| `min_neighbors` | 10 | Seuil du filtre de densité |
| `neighbor_radius` | 0.25 | Rayon de recherche KDTree (m) |
| `buffer_frames` | 5 | Taille du buffer temporel |
| `min_confirmations` | 3 | Min images pour confirmation |
| `bin_size` | 0.3 | Taille du bin spatial (m) |
| `dbscan_eps` | 0.15 | Voisinage DBSCAN (m) |
| `dbscan_min_samples` | 130 | Seuil cœur DBSCAN |
| `dbscan_min_cluster_size` | 120 | Taille min du cluster |

**Note :** Les variantes de filtre ont été supprimées au profit de ce pipeline unifié.

### 2.3 Nœud 02 : Extraction des Bordures d’Anneaux (v3)
| Propriété | Valeur |
|-----------|--------|
| **Fichier** | `02_ring_boundary_extraction.py` |
| **Entrée** | `/road_surface` |
| **Sortie** | `/ring_boundaries` (PointCloud2) |
| **Algorithme** | Extrémum par anneau + ajustement de courbe RANSAC + contraintes |

**Paramètres clés :**
| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `max_rings` | 15 | Limite anneaux 0-14 |
| `min_points_per_ring` | 50 | Anneau valide minimum |
| `robust_n_points` | 3 | Médiane des N extrêmes |
| `enable_ring_consistency` | true | Propagation des voisins |
| `max_y_deviation` | 0.5 | Saut max entre anneaux |
| `temporal_buffer_frames` | 5 | Moyenne temporelle pondérée |
| `enable_curve_fitting` | true | Ajustement RANSAC/spline |
| `fitting_method` | ransac | Options : ransac/spline/both |
| `polynomial_degree` | 2 | Ajustement parabolique |
| `ransac_residual_threshold` | 0.15 | Seuil d’outlier (m) |
| `min_points_for_fitting` | 10 | Points min pour ajustement |
| `curve_resampling_spacing` | 0.5 | Espacement uniforme des points (m) |
| `enforce_constant_width` | true | Contrainte largeur de route |
| `width_tolerance` | 0.1 | Variation largeur 10% |
| `enforce_symmetry` | false | Forcer symétrie autour de l’axe |

**Note :** L’ajustement polynomial RANSAC a été supprimé. L’ajustement de courbe est maintenant géré dans le Nœud 02.

---

## 3. Référence des Topics

| Topic | Type | Hz | Publisher | Subscriber |
|-------|------|-----|-----------|------------|
| `/velodyne_points` | PointCloud2 | 10 | Driver LiDAR | Patchwork++ |
| `/patchworkpp/ground` | PointCloud2 | 10 | Patchwork++ | Nœud 01 |
| `/road_surface` | PointCloud2 | 10 | Nœud 01 | Nœud 02 |
| `/ring_boundaries` | PointCloud2 | 10 | Nœud 02 | Nœud 03 |
| `/road_boundaries` | MarkerArray | 10 | Nœud 03 | RViz2 |

**QoS :** Tous les topics utilisent la fiabilité `BEST_EFFORT`, la durabilité `VOLATILE`.

---

## 4. Structure des Fichiers

```
src/drivable_area_detector/
├── ARCHITECTURE.md                    # Ce fichier
├── config/
│   └── params.yaml                    # Tous les paramètres des nœuds
├── drivable_area_detector/
│   ├── 01_road_points_extraction.py   # Nœud 01
│   ├── 02_ring_boundary_extraction.py # Nœud 02
│   ├── 03_roundabout_handler.py       # RÉSERVÉ (non lancé)
│   ├── ransac_fitting.py              # Utilitaires RANSAC
│   ├── rviz2_publisher.py             # Nœud 03 (marqueurs)
│   └── geometry_utils.py              # Fonctions géométriques partagées
├── launch/
│   └── detector.launch.py             # Lancement des 3 nœuds
└── test/
    └── ...
```

---

## 5. Visualisation RViz2

**Types de marqueurs :**
- `LINE_STRIP` (ns : `road_boundaries`, id=0) : Ligne de bordure gauche
- `LINE_STRIP` (ns : `road_boundaries`, id=1) : Ligne de bordure droite

**Note :** La zone de route remplie (TRIANGLE_LIST) a été supprimée dans la refonte "Less is More".

**Couleurs :**
| Élément | RGB | Description |
|---------|-----|-------------|
| Bordure gauche | Cyan (0,1,1) | Côté conducteur gauche |
| Bordure droite | Magenta (1,0,1) | Côté conducteur droite |

**Frame :** `velodyne` (pas `pandora`)

---

## 6. Outils & Réglages

| Outil | But | Utilisation |
|-------|-----|-------------|
| `boundary_extraction_tuner.py` | Régler les paramètres du Nœud 02 | `python3 tools/boundary_extraction_tuner.py --latest` |
| `detection_validator.py` | Validation image par image | `make validate-detection NODE=02` |
| `marker_array_tuner.py` | Régler les params RANSAC du Nœud 03 | `python3 tools/marker_array_tuner.py --latest` |

**Workflow de validation :**
1. Exécuter `make validate-detection NODE=XX` pour labelliser les images bonnes/mauvaises
2. Utiliser l’outil tuner pour trouver les paramètres optimaux
3. Mettre à jour `params.yaml` avec les recommandations
4. Recompiler et tester

---

## 7. Points d’Extension

### Modifier les paramètres DBSCAN
1. Modifier `params.yaml` → section `road_points_extraction_node`
2. Paramètres clés : `dbscan_eps`, `dbscan_min_samples`, `dbscan_min_cluster_size`
3. Tester avec `detection_validator.py --node 01`

### Modifier l’ajustement de courbe (Nœud 02)
1. Modifier `params.yaml` → section `ring_boundary_node`
2. Options : `fitting_method` (ransac/spline/both), `polynomial_degree`
3. Tester avec `detection_validator.py --node 02`

### Intégration du gestionnaire de rond-point
Le fichier réservé `03_roundabout_handler.py` peut être activé en :
1. Ajoutant le point d’entrée dans `setup.py`
2. Ajoutant le nœud dans `detector.launch.py`
3. Insérant entre Nœud 02 et Nœud 03

---

*Dernière mise à jour : 2026-01-13*

