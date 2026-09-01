# Détecteur de Zone Circulable

Package ROS2 pour la détection en temps réel de la zone circulable à partir de données LiDAR. Ce package traite les nuages de points segmentés pour extraire les bordures de route et visualiser la région navigable pour les véhicules autonomes.

## Vue d'Ensemble

Le package `drivable_area_detector` implémente un pipeline robuste en 3 étapes pour détecter et visualiser les bordures de route à partir de données LiDAR :

1. **Extraction de la Surface Routière** - Filtre les points au sol pour identifier la surface asphaltée en utilisant un filtrage par intensité et un clustering DBSCAN
2. **Extraction des Bordures par Anneau** - Extrait les bordures gauche/droite à partir des anneaux LiDAR avec un ajustement de courbe RANSAC
3. **Visualisation** - Publie des marqueurs de bordures pour la visualisation dans RViz2

## Fonctionnalités

- **Détection de route basée sur l'intensité** avec confirmation temporelle pour plus de robustesse
- **Clustering DBSCAN** pour la réduction du bruit et l'élimination des valeurs aberrantes
- **Extraction de bordures par anneau** utilisant la structure en anneaux du LiDAR (support des anneaux 0-14)
- **Ajustement de courbe polynomiale RANSAC** (degré 2) pour une estimation lisse des bordures
- **Contraintes géométriques** incluant la cohérence de largeur et l'application de la symétrie
- **Visualisation en temps réel** avec marqueurs de bordures colorés (cyan/magenta)
- **Paramètres configurables** via fichier de configuration YAML
- **Lissage temporel** avec buffers de trames pondérés

## Architecture

```
/patchworkpp/ground (PointCloud2)
       │
       ▼
┌─────────────────────────────────────────┐
│ [Nœud 01] Extraction Points Route       │
│  • Filtre intensité : [0.0, 8.0]        │
│  • Validation densité (cKDTree)         │
│  • Confirmation temporelle (5 trames)   │
│  • Clustering DBSCAN                    │
└─────────────────────────────────────────┘
       │ /road_surface (PointCloud2)
       ▼
┌─────────────────────────────────────────┐
│ [Nœud 02] Extraction Bordures Anneaux   │
│  • Détection extremum par anneau        │
│  • Ajustement courbe RANSAC (degré 2)   │
│  • Contraintes largeur & lissage        │
│  • Moyenne temporelle (5 trames)        │
└─────────────────────────────────────────┘
       │ /ring_boundaries (PointCloud2)
       ▼
┌─────────────────────────────────────────┐
│ [Nœud 03] Éditeur RViz2                 │
│  • Marqueurs LINE_STRIP                 │
│  • Cyan (gauche) + Magenta (droite)     │
└─────────────────────────────────────────┘
       │ /road_boundaries (MarkerArray)
       ▼
     RViz2
```

## Prérequis

### Dépendances ROS2
- ROS2 (Humble ou ultérieur recommandé)
- `rclpy`
- `sensor_msgs`
- `sensor_msgs_py`
- `visualization_msgs`
- `tf2_ros`
- `tf2_sensor_msgs`

### Dépendances Python
- `numpy` - Opérations numériques
- `scipy` - Calcul scientifique et traitement du signal
- `scikit-learn` - Clustering DBSCAN et RANSAC
- `scikit-image` - Utilitaires de traitement d'image
- `open3d` - Traitement de nuages de points 3D
- `pyyaml` - Chargement de paramètres YAML

### Nœuds Externes
- **Patchwork++** - Nœud de segmentation du sol (publie `/patchworkpp/ground`)

## Installation

### 1. Cloner dans l'Espace de Travail ROS2

```bash
cd ~/ros2_ws/src
git clone <url-du-dépôt> drivable_area_detector
```

### 2. Installer les Dépendances Python

```bash
pip install numpy scipy scikit-learn scikit-image open3d pyyaml
```

Ou utiliser rosdep :

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Compiler le Package

```bash
cd ~/ros2_ws
colcon build --packages-select drivable_area_detector
source install/setup.bash
```

## Utilisation

### Lancer le Pipeline Complet

```bash
ros2 launch drivable_area_detector detector.launch.py
```

Ceci lance les trois nœuds avec les paramètres par défaut depuis `config/params.yaml`.

### Lancer les Nœuds Individuellement

**Nœud 01 - Extraction des Points de Route :**
```bash
ros2 run drivable_area_detector 01_road_points_extraction_node
```

**Nœud 02 - Extraction des Bordures par Anneau :**
```bash
ros2 run drivable_area_detector 02_ring_boundary_extraction_node
```

**Nœud 03 - Éditeur de Visualisation :**
```bash
ros2 run drivable_area_detector road_markers_publisher
```

### Visualiser dans RViz2

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix drivable_area_detector)/share/drivable_area_detector/config/drivable_area.rviz
```

## Configuration

Tous les paramètres sont définis dans `config/params.yaml` :

### Paramètres du Nœud 01 (Extraction Points Route)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `min_road_intensity` | 0.0 | Intensité minimale pour la détection d'asphalte |
| `max_road_intensity` | 6.0 | Intensité maximale pour la détection d'asphalte |
| `max_range` | 25.0 | Distance maximale du véhicule (m) |
| `min_neighbors` | 10 | Nombre minimal de voisins pour le filtre de densité |
| `neighbor_radius` | 0.25 | Rayon de recherche des voisins (m) |
| `buffer_frames` | 5 | Taille du buffer temporel |
| `min_confirmations` | 3 | Nombre minimal de trames pour confirmation |
| `bin_size` | 0.3 | Taille du bin spatial (m) |
| `dbscan_eps` | 0.15 | Rayon de voisinage DBSCAN (m) |
| `dbscan_min_samples` | 130 | Seuil de point noyau DBSCAN |
| `dbscan_min_cluster_size` | 120 | Taille minimale du cluster |

### Paramètres du Nœud 02 (Extraction Bordures Anneaux)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `max_rings` | 15 | Nombre maximal d'anneaux LiDAR (0-14) |
| `min_points_per_ring` | 50 | Points minimaux pour anneau valide |
| `robust_n_points` | 3 | Nombre de points pour calcul médiane |
| `enable_ring_consistency` | true | Activer la propagation des voisins |
| `max_y_deviation` | 0.5 | Déviation max entre anneaux (m) |
| `temporal_buffer_frames` | 5 | Taille du buffer de moyenne temporelle |
| `enable_curve_fitting` | true | Activer l'ajustement de courbe RANSAC |
| `fitting_method` | ransac | Méthode d'ajustement (ransac/spline/both) |
| `polynomial_degree` | 2 | Degré polynomial pour l'ajustement |
| `ransac_residual_threshold` | 0.15 | Seuil d'outlier RANSAC (m) |
| `min_points_for_fitting` | 10 | Points minimaux pour ajustement de courbe |
| `curve_resampling_spacing` | 0.5 | Espacement uniforme des points (m) |
| `enforce_constant_width` | true | Appliquer les contraintes de largeur |
| `width_tolerance` | 0.1 | Tolérance de variation de largeur (10%) |
| `enforce_symmetry` | false | Forcer la symétrie autour de l'axe central |

### Modifier les Paramètres

1. Éditer `config/params.yaml`
2. Recompiler le package : `colcon build --packages-select drivable_area_detector`
3. Re-sourcer : `source install/setup.bash`
4. Relancer les nœuds

## Topics ROS2

### Topics Souscrits

| Topic | Type | Description |
|-------|------|-------------|
| `/patchworkpp/ground` | `sensor_msgs/PointCloud2` | Nuage de points segmenté au sol depuis Patchwork++ |
| `/road_surface` | `sensor_msgs/PointCloud2` | Points de surface routière (souscrit par Nœud 02) |
| `/ring_boundaries` | `sensor_msgs/PointCloud2` | Points de bordure par anneau (souscrit par Nœud 03) |

### Topics Publiés

| Topic | Type | Hz | Description |
|-------|------|-------|-------------|
| `/road_surface` | `sensor_msgs/PointCloud2` | 10 | Points de surface routière filtrés |
| `/ring_boundaries` | `sensor_msgs/PointCloud2` | 10 | Points de bordure gauche/droite par anneau |
| `/road_boundaries` | `visualization_msgs/MarkerArray` | 10 | Marqueurs de lignes de bordure pour RViz2 |

### Paramètres QoS

Tous les topics utilisent :
- **Fiabilité :** `BEST_EFFORT`
- **Durabilité :** `VOLATILE`

## Visualisation

L'éditeur RViz2 crée des marqueurs LINE_STRIP pour les bordures de route :

| Élément | Couleur | ID Marqueur | Description |
|---------|---------|-------------|-------------|
| Bordure Gauche | Cyan (0, 1, 1) | 0 | Bord gauche de la route |
| Bordure Droite | Magenta (1, 0, 1) | 1 | Bord droit de la route |

**Frame :** Tous les marqueurs sont publiés dans le référentiel `velodyne`.

## Développement & Tests

### Exécuter les Tests

```bash
cd ~/ros2_ws
colcon test --packages-select drivable_area_detector
colcon test-result --verbose
```

### Suites de Tests Disponibles

- `test_road_extraction.py` - Validation de l'extraction des points de route
- `test_boundary_detection.py` - Tests d'extraction de bordures
- `test_ransac_fitting.py` - Tests d'ajustement de courbe RANSAC
- `test_noise_filtering.py` - Validation des filtres de bruit
- `test_roundabout_handler.py` - Tests de gestion des ronds-points
- `test_validation_data.py` - Validation des jeux de données
- Tests standards : `test_flake8.py`, `test_pep257.py`, `test_copyright.py`

### Outils de Réglage des Paramètres

Pour des workflows de réglage détaillés, consulter `ARCHITECTURE.md` :

- Valider la qualité de détection sur des données enregistrées
- Ajuster les paramètres DBSCAN et RANSAC
- Optimiser pour des conditions de route spécifiques

## Structure du Projet

```
drivable_area_detector/
├── ARCHITECTURE.md                     # Documentation détaillée de l'architecture
├── README.md                           # Ce fichier
├── package.xml                         # Manifeste du package ROS2
├── setup.py                           # Configuration du package Python
├── setup.cfg                          # Configuration du package
│
├── config/
│   ├── params.yaml                    # Tous les paramètres des nœuds
│   └── drivable_area.rviz            # Configuration RViz2
│
├── drivable_area_detector/            # Code principal du package
│   ├── __init__.py
│   ├── 01_road_points_extraction.py   # Nœud 01 : Extraction surface route
│   ├── 02_ring_boundary_extraction.py # Nœud 02 : Extraction bordures
│   ├── 03_roundabout_handler.py       # Réservé (non lancé)
│   └── rviz2_publisher.py             # Nœud 03 : Visualisation
│
├── launch/
│   └── detector.launch.py             # Fichier de lancement pour tous les nœuds
│
├── test/                              # Suite de tests
│   ├── test_road_extraction.py
│   ├── test_boundary_detection.py
│   ├── test_ransac_fitting.py
│   └── ...
│
└── resource/                          # Fichiers ressources ROS2
    └── drivable_area_detector
```

## Composants Réservés

- **`03_roundabout_handler.py`** - Détection et gestion des ronds-points (non actuellement lancé)

Ce nœud peut être intégré en l'ajoutant aux points d'entrée dans `setup.py` et dans `detector.launch.py`.

## Dépannage

### Pas de sortie sur `/road_surface`

- Vérifier que `/patchworkpp/ground` publie correctement
- Vérifier que la plage d'intensité correspond à votre LiDAR (ajuster `min_road_intensity`, `max_road_intensity`)
- Réduire `dbscan_min_samples` si la route est très clairsemée

### Détection de bordures bruitée

- Augmenter `ransac_residual_threshold` pour des courbes plus lisses
- Augmenter `min_points_per_ring` pour ignorer les anneaux peu fiables
- Activer `enforce_constant_width` pour une largeur plus stable

### Bordures qui sautent entre les trames

- Augmenter `temporal_buffer_frames` pour plus de lissage temporel
- Activer `enable_ring_consistency` pour le lissage spatial
- Ajuster `max_y_deviation` pour autoriser/restreindre les sauts anneau-à-anneau

## Licence

Licence MIT

## Mainteneur

**Victor Dessenne**
Email : 83450047+vdessenn@users.noreply.github.com

## Version

1.0.0

## Voir Aussi

- **ARCHITECTURE.md** - Architecture détaillée du système et descriptions des algorithmes
- **config/params.yaml** - Référence complète des paramètres avec descriptions
