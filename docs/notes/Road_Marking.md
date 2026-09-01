# Road Marking Detection Using LIDAR Reflective Intensity Data and its Application to Vehicle Localization  
_A. Hata, D. Wolf – ITSC 2014_

## 1. Objectif et contexte

L’article propose une méthode de détection de marquages routiers (lignes de voie, passages piétons, etc.) à partir de l’intensité de réflexion renvoyée par un LIDAR multilignes, et montre comment ces marquages peuvent être utilisés comme éléments de carte pour la localisation d’un véhicule autonome. La motivation principale est de s’affranchir des limites des caméras (sensibles à l’illumination) en exploitant l’intensité infrarouge quasi indépendante de la lumière ambiante. 

## 2. Pipeline global

Le pipeline proposé se décompose en trois blocs principaux :  
1. **Détection des bordures de route (curbs)** pour délimiter la zone de recherche des marquages.  
2. **Calibration de l’intensité LIDAR** pour corriger les biais de réponse des différents faisceaux.  
3. **Segmentation des marquages routiers** par un Otsu modifié, puis utilisation des marquages comme entités de carte pour la localisation Monte Carlo. 

---

## 3. Détection de bordures (curbs)

### 3.1 Idée générale

Les données du LIDAR multilignes sont vues comme des « anneaux » (rings) correspondant à chaque couche verticale. Sur une surface plane, la distance radiale de l’anneau \(i\) vaut :

\[
r_i = h \cot(\theta_i)
\]

où \(h\) est la hauteur du capteur et \(\theta_i\) l’angle d’élévation du faisceau \(i\). 

La « compression » entre deux anneaux consécutifs est :

\[
\Delta r_i = r_{i+1} - r_i = h \left( \cot(\theta_{i+1}) - \cot(\theta_i) \right)
\]

Sur une route plane, les \(\Delta r_i\) restent autour d’une valeur nominale ; lorsqu’un faisceau rencontre un obstacle (trottoir), la compression sort d’un intervalle attendu. 

### 3.2 Détection dans une grille circulaire

- Les retours LIDAR sont stockés dans une grille polaire (anneaux \(i\), cellules angulaires \(j\)).  
- Chaque cellule contient la distance moyenne \(d_{i,j}\) et la position moyenne \((x_{i,j}, y_{i,j}, z_{i,j})\).  
- Une cellule est candidate « curb » si la différence de distance entre anneaux successifs est dans un intervalle \(I_i\) :

\[
| d_{i+1,j} - d_{i,j} | \in I_i = [\alpha \Delta r_i,\ \beta \Delta r_i]
\]

avec \(\beta > \alpha\). 

Pour réduire les faux positifs (véhicules, obstacles divers), trois filtres sont enchaînés :  
1. **Filtre différentiel** : convolution locale pour mesurer la pente (steepness).  
2. **Filtre de distance** : on ne conserve que les obstacles les plus proches, compatibles avec la position des trottoirs.  
3. **Filtre de régression robuste** (Least Trimmed Squares) : ajustement d’un modèle de bordure, rejetant les cellules incohérentes. 

Le résultat est une estimation des bordures gauche et droite, utilisées ensuite pour borner la zone de recherche des marquages. 

---

## 4. Calibration de l’intensité LIDAR

### 4.1 Problème

À cause des variations de fabrication, un même matériau (asphalte, peinture) peut renvoyer des intensités différentes selon le faisceau et la distance. Sans calibration, un seuil global sur l’intensité risque d’échouer. 

### 4.2 Méthode déterministe (résumé)

1. Collecter des nuages de points LIDAR sur un environnement arbitraire, avec poses du véhicule.  
2. Projeter les points dans une grille 2D au sol ; chaque cellule contient une liste de points.  
3. Pour chaque paire (ID de faisceau \(j\), intensité brute \(a\)) :  
   - Chercher toutes les cellules contenant ce couple.  
   - Calculer la moyenne des intensités dans ces cellules, en excluant les retours provenant du même faisceau \(j\).  
   - Définir l’intensité calibrée de \((j, a)\) comme cette moyenne. 

On obtient ainsi une table de calibration \(32 \times 256\) (pour 32 faisceaux HDL‑32E et 256 niveaux d’intensité), qui permet de corriger chaque point en ligne : intensité calibrée = table[faisceau][intensité brute]. 

---

## 5. Détection des marquages par Otsu modifié

### 5.1 Segmentation intensité asphalte / peinture

Dans la zone délimitée par les bordures \(f_l(x)\) et \(f_r(x)\), on ne considère que les points dont la coordonnée latérale \(y\) est comprise entre ces courbes (surface de route). 

En pratique, l’histogramme d’intensité des points route est bimodal :  
- un mode pour l’asphalte,  
- un mode pour les marquages (peinture plus réfléchissante ou moins, selon le capteur et la calibration). 

L’algorithme s’appuie sur Otsu, en le complétant de tests supplémentaires.

### 5.2 Otsu + critères de robustesse

Soit \(n_i\) le nombre de points ayant l’intensité \(i\), \(M\) le nombre total de points route, et \(L\) le nombre de niveaux possibles (256 pour un octet) :

1. **Histogramme normalisé**  
\[
p_i = \frac{n_i}{M}
\]

2. **Somme cumulée**  
\[
P(k) = \sum_{i=0}^{k} p_i
\]

3. **Moyenne cumulée**  
\[
m(k) = \sum_{i=0}^{k} i\, p_i
\]

4. **Moyenne globale**  
\[
m_G = \sum_{i=0}^{L-1} i\, p_i
\]

5. **Variance globale**  
\[
\sigma_G^2 = \sum_{i=0}^{L-1} (i - m_G)^2 p_i
\]

6. **Variance inter-classes (Otsu)**  
\[
\sigma_L^2(k) = \frac{ \left( m_G P(k) - m(k) \right)^2 }{P(k)\left(1 - P(k)\right)}
\]

7. **Seuil optimal brut**  
\[
T = \arg\max_{0 \le k \le L-1} \sigma_L^2(k)
\]

8. **Mesure de séparabilité**  
\[
\eta(T) = \frac{\sigma_L^2(T)}{\sigma_G^2}
\]

On impose \(\eta(T) \ge t_\eta\), avec \(t_\eta\) proche de 1 pour garantir une bonne séparation. 

9. **Contraintes supplémentaires sur \(T\)**  
- \(T\) doit être compatible avec l’intensité attendue des marquages (par exemple \(T \ge t_T\)).  
- La somme cumulée \(P(T)\), qui reflète la proportion de points sous le seuil, ne doit pas dépasser un seuil \(t_P\) (les marquages ne couvrent qu’une fraction de la route). 

10. **Classification des points**  
Un point route est classé « marquage » si son intensité calibrée est du côté prévu de \(T\) (selon convention) et si les critères de séparabilité sont satisfaits. 

L’algorithme recalcule \(T\) dynamiquement pour chaque acquisition afin de s’adapter à l’usure et aux variations de peinture et d’asphalte. 

11. **Filtrage de segments trop longs**  
Pour limiter les faux positifs, les séquences contiguës de points « marquage » qui dépassent une longueur maximale (en largeur) sont rejetées, car les marquages typiques (lignes, passages piétons) ont une largeur bornée. 

---

## 6. Utilisation pour la localisation (MCL)

### 6.1 Construction de la carte

- Les nuages de points contenant **curbs + marquages** sont alignés hors-ligne par une méthode SLAM de type ELCH (Explicit Loop Closing Heuristic).  
- On obtient une carte d’occupation 2D avec une résolution d’environ 0,10 m.  
- Les points de bordure et de marquage sont intégrés comme « obstacles » structurés dans la carte. 

### 6.2 Monte Carlo Localization

- Un filtre de particules (MCL) utilise la carte et les observations LIDAR.  
- Les particules sont initialisées autour de la position GPS (avec incertitude).  
- À chaque pas, les particules sont pondérées selon la cohérence entre les observations (curbs + marquages détectés) et la carte.  
- Les particules avec meilleure correspondance sont resélectionnées ; la meilleure particule donne la pose estimée. 

Deux variantes sont comparées :  
1. MCL avec **seulement les curbs** comme entités de carte.  
2. MCL avec **curbs + marquages routiers**. 

---

## 7. Résultats expérimentaux (résumé chiffré)

Les expériences sont réalisées sur un véhicule autonome (CaRINA 2) équipé d’un Velodyne HDL‑32E, sur un parcours urbain d’environ 822 m, avec odométrie/GPS fournie par un Xsens MTi‑G. 

En moyenne :  
- **Erreur latérale MCL (y)** :  
  - Curb seul : ~0,58 m  
  - Curb + marquages : ~0,31 m  
  → réduction d’environ 54 % de l’erreur latérale. 

- **Erreur longitudinale MCL (x)** :  
  - Curb seul : ~1,32 m  
  - Curb + marquages : ~1,20 m  
  → légère amélioration longitudinale. 

- **Variances MCL** :  
  - Variance latérale et longitudinale réduites respectivement d’environ 71 % et 60 % avec les marquages. 

Ces résultats montrent que l’ajout des marquages comme entités de carte renforce significativement la précision latérale et la confiance de la localisation dans des rues urbaines souvent symétriques. 

---

## 8. Points clés à retenir

- L’intensité LIDAR (infrarouge) permet une détection robuste des marquages, indépendante de la lumière visible.   
- La calibration d’intensité au niveau faisceau × niveau de gris est cruciale pour rendre le seuil Otsu fiable.   
- L’Otsu modifié (avec contraintes sur séparabilité, position du seuil et somme cumulée) fournit un seuil adaptatif robuste pour séparer asphalte et peinture.   
- Les marquages intégrés aux curbs comme « features » de carte améliorent nettement la localisation Monte Carlo, surtout sur l’axe latéral. 

---

## 9. Référence bibliographique

A. Hata, D. Wolf, “Road Marking Detection Using LIDAR Reflective Intensity Data and its Application to Vehicle Localization,” _IEEE Intelligent Transportation Systems Conference (ITSC)_, 2014. 
