# Pandora – Résumé données LiDAR

## Vue d’ensemble

- LiDAR mécanique rotatif 40 canaux, balayage 360° horizontal, FOV vertical de −16° à +7°.  
- Portée de 0,3 m à 200 m sur cible 20 % réflectivité, précision typique de quelques centimètres.  
- Fréquences de trame : 10 Hz ou 20 Hz, jusqu’à 720 k points/s.  
- Modes de retour : dernier retour, retour le plus fort, double retour.  
- Sortie sur Ethernet Gigabit, protocole UDP/IP.

---

## Spécifications angulaires et canaux

- 40 canaux verticaux, chacun avec un angle d’élévation fixe couvrant de −16° à +7°.  
- Résolution verticale :  
  - 0,33° entre les canaux « centraux » (zone densifiée).  
  - 1° dans les zones hautes et basses.  
- Azimut d’un point = azimut de bloc (angle du rotor) + offset horizontal propre au canal.  
- Offsets horizontaux/verticaux par canal fournis dans la table de distribution des 40 canaux (Appendix I).

---

## Paquet UDP LiDAR (point cloud)

- Transport : Ethernet Gigabit, UDP, port source typique 10000, port destination typique 2368.  
- Structure d’un paquet point cloud :  
  - En-tête Ethernet+IP+UDP : 42 octets.  
  - Charge utile UDP : 1262 octets.  

### Structure de la charge utile (1262 octets)

- 1240 octets de « Ranging Data ».  
- 22 octets d’« Additional Information ».

#### Ranging Data (1240 octets)

- 10 blocs consécutifs de 124 octets chacun.  
- Par bloc :  
  - 2 octets de header fixe (mot de synchronisation, par exemple 0xFFEE).  
  - 2 octets d’azimut (angle de référence rotor, généralement en centièmes de degré, little-endian).  
  - 40 « units » de mesure de 3 octets :  
    - Distance : 2 octets non signés, résolution 4 mm.  
    - Intensité / réflectivité : 1 octet.  

- En mode double retour :  
  - Les blocs vont par paires (1 & 2, 3 & 4, …) pour le même ensemble de 40 tirs.  
  - Le bloc impair (1, 3, 5, …) correspond en général au « last return ».  
  - Le bloc pair (2, 4, 6, …) correspond au « strongest return ».  
  - L’azimut ne change que toutes les deux trames de bloc.

---

## Champ « Additional Information » (22 octets)

- 8 octets : réservés (sans impact sur le parsing standard).  
- 2 octets : vitesse moteur (rpm, avec un facteur d’échelle indiqué dans la doc).  
- 4 octets : GPS Timestamp (fraction de seconde entre 0 et 1 s, base de temps commune avec les paquets GPS).  
- 1 octet : mode de retour (codes distincts pour strongest, last, dual).  
- 1 octet : information usine (identification de version / usine).  
- 6 octets : temps UTC (année, mois, jour, heure, minute, seconde, codés en décimal).

---

## Paquet UDP GPS (synchronisation temporelle)

- Transport : UDP, port destination typique 10110.  
- Structure :  
  - En-tête Ethernet+IP+UDP : 42 octets.  
  - Données UDP : 512 octets.  

### Contenu des 512 octets

- 18 octets de « GPS Time Data » :  
  - Mot de synchro (0xFFEE).  
  - Date : 6 octets ASCII (YYMMDD).  
  - Heure : 6 octets ASCII (HHMMSS).  
  - Champ sTime : 4 octets, fraction de seconde (0–1 s), même source que le timestamp des paquets LiDAR.  
- 77 octets : trame NMEA GPRMC (ASCII complète).  
- Reste : octets réservés (dont indicateurs de validité, lock PPS, etc.).

### Génération des paquets GPS

- À chaque front montant du PPS GPS, le module envoie un paquet GPS.  
- L’UTC des paquets GPS est ajusté pour correspondre au temps exact du front PPS.  
- Si la trame GPRMC n’est pas disponible à cet instant, la dernière GPRMC est réutilisée avec une logique d’ajustement interne.

---

## Calcul du temps absolu d’un paquet point cloud

On cherche à obtenir un timestamp absolu \(t_0\) pour chaque paquet LiDAR.

### Méthode 1 : autonome (point cloud seul)

1. Lire le champ GPS Timestamp (4 octets) dans les 22 octets « Additional Information ».  
2. Lire les 6 octets UTC dans le même paquet.  
3. Construire l’instant absolu \(t_0\) :  
   - UTC (année, mois, jour, heure, minute, seconde) comme partie entière des secondes.  
   - Ajouter la fraction de seconde issue du GPS Timestamp.

### Méthode 2 : via le dernier paquet GPS

1. Récupérer le timestamp de fraction de seconde dans le paquet point cloud.  
2. Conserver le dernier paquet GPS reçu et son UTC.  
3. Combiner fraction de seconde (du point cloud) + UTC (du dernier GPS) pour obtenir \(t_0\).  

Les deux méthodes donnent le même résultat lorsque les paquets sont correctement synchronisés.

---

## Temps de tir des lasers à l’intérieur d’un paquet

- Chaque paquet point cloud contient 10 blocs.  
- La documentation fournit des formules explicites pour :  
  - Le temps de fin de chaque bloc en fonction de \(t_0\) (temps de « packing » du paquet) et de décalages fixes de l’ordre de dizaines de microsecondes.  
  - Les temps de tir individuels pour chacun des 40 canaux dans un bloc, via une liste ordonnée d’offsets par ID laser.

Principe général :

1. Calculer \(t_0\), le temps absolu de « fin » du paquet.  
2. Appliquer les offsets fournis pour obtenir l’heure de fin de chaque bloc (10 valeurs).  
3. Pour un bloc donné, appliquer les offsets propres à chaque canal pour obtenir l’instant de tir individuel du laser.

En mode double retour :

- Les blocs 1 & 2, 3 & 4, etc. représentent le même ensemble de tirs mais avec deux sélections de retour (last / strongest).  
- Les instants de tir laser pour les deux blocs d’une même paire sont identiques, seuls les champs distance/intensité changent.

---

## Reconstruction géométrique d’un point

Pour chaque « unit » (distance + intensité) :

1. Récupérer l’azimut du bloc (angle rotor) et l’offset horizontal du canal ; sommer les deux pour obtenir l’angle horizontal.  
2. Prendre l’élévation fixe du canal (table des 40 canaux).  
3. Convertir la distance (valeur brute × 4 mm) en mètres.  
4. Projeter en coordonnées cartésiennes (x, y, z) dans le repère du LiDAR :  
   - \(x = d \cos(\text{elev}) \cos(\text{az})\)  
   - \(y = d \cos(\text{elev}) \sin(\text{az})\)  
   - \(z = d \sin(\text{elev})\)  
5. Associer le timestamp du tir (calculé via \(t_0\) + offsets bloc/canal) et l’intensité de réflectivité.

---

## Résumé exploitable pour un parser

Un parser LiDAR Pandora devra :

- Écouter les ports UDP des paquets point cloud et GPS.  
- Décoder les en-têtes Ethernet/IP/UDP pour filtrer les bonnes trames.  
- Pour chaque paquet point cloud :  
  - Parcourir les 10 blocs, extraire azimut + 40 units (distance, intensité).  
  - Lire les 22 octets d’« Additional Information » pour le timestamp et le mode de retour.  
  - Calculer \(t_0\) (méthode 1 ou 2).  
  - Calculer les instants de tir par bloc et par canal avec les offsets fournis.  
  - Appliquer la table des canaux (offset horizontal + élévation) pour chaque ID canal.  
  - Convertir en \((x,y,z,t,\text{intensité},\text{canal},\text{mode de retour})\).  
- Utiliser les paquets GPS pour :  
  - Suivi de l’UTC exact.  
  - Vérification de la validité (flag lock PPS, statut GPRMC).

---

## Format des pointclouds publiés (ROS2 PointCloud2)

Le LiDAR Pandora publie les nuages de points au format standard ROS2 `sensor_msgs/PointCloud2`. Ce format est utilisé dans la plupart des systèmes robotiques pour l’échange et le traitement des données 3D.

### Type de message
- **Type ROS2** : `sensor_msgs/PointCloud2`
- **Documentation officielle** : [sensor_msgs/PointCloud2](https://docs.ros.org/en/foxy/api/sensor_msgs/msg/PointCloud2.html)

### Champs principaux du message
Chaque point du nuage contient les informations suivantes :

| Champ      | Description technique                          | Type      |
|------------|-----------------------------------------------|-----------|
| x          | Coordonnée X (mètres)                         | float32   |
| y          | Coordonnée Y (mètres)                         | float32   |
| z          | Coordonnée Z (mètres)                         | float32   |
| intensity  | Intensité de réflectivité du tir laser        | float32   |
| ring       | Index du canal (0 à 39 pour Pandora)          | uint16    |
| time       | Timestamp relatif du tir (secondes)           | float32   |

Autres métadonnées du message :
- **header.frame_id** : nom de la frame de référence (ex : `velodyne`)
- **header.stamp** : timestamp ROS du message
- **height, width** : dimensions du tableau de points (souvent height=1, width=nombre de points)
- **is_dense** : indique la présence de points invalides

### Organisation des données
- Les points sont ordonnés selon les tirs laser et la séquence de balayage.
- Le champ `ring` permet d’identifier le canal vertical associé à chaque point.
- Le champ `time` correspond au temps relatif du tir par rapport au début de la trame, utile pour la synchronisation fine.
- L’intensité renseigne sur la réflectivité de la cible.

### Exemple de structure de point
```yaml
- x: 12.34
  y: 56.78
  z: 9.10
  intensity: 42.0
  ring: 17
  time: 0.00234
```

### Référence
Pour plus de détails sur le format et l’utilisation du message : [ROS2 sensor_msgs/PointCloud2](https://docs.ros.org/en/foxy/api/sensor_msgs/msg/PointCloud2.html)
