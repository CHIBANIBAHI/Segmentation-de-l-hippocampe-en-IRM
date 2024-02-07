# Segmentation-de-l-hippocampe-en-IRM
Membres du projet:

FIRDAOUISSI Ouafae     et      CHIBANI BAHI Ouissem

DESCRIPTION DU PROJET:

Les recherches scientifiques ont démontré qu'il existait des marqueurs en IRM (imagerie par résonance magnétique) cérébral des états de stress post-traumatiques (ESPT) tandis que d'autres recherches ont montré que le volume de l'hippocampe différait selon que les fonctions cognitives des participants étaient normales ou diminuées.

Dans ce projet nous envisagerons en premier lieu de faire des recherches sur le contexte : le cerveau, l'IRM cérébral..., ensuite nous entamons des recherches bibliographiques sur les marqueurs des IRM cérébraux en ESPT, ainsi que pour les maladies où les fonctions cognitives sont touchées  (maladie d'Alzheimer, skizophrène..).Enfin, nous attaquons la partie de la réalisation d'une segmentation automatique par réseaux de neurones de l'hippocampe sur une autre base de données d'IRM existante et libre de droit (patients schizophrènes ou Alzheimer), du calcul du volume hippocampique et de la comparaison des résultats entre sujets contrôle et patients.

# Synthèse d'avancement du projet jusqu'à présent:

I- Etat de l’art sur:

Hippocampe et PTSD:

D’après les articles sélectionnés et étudiés (Sonalee A. Joshi et al 2019, hahin Zandieh et al 2016, Ruth Klaming et al 2019,
J. Douglas Bremner et al 1997, J. Douglas Bremner et al 1995b, J. Douglas Bremner 2006, B. R.Filipovic , et al 2011, Philip R 2018, Ana Starcevic et al 2014), le volume hippocampique se réduit lors d’une atteinte du syndrome du stress post-traumatique (SSPT) quelque soit la cause: (torture, ELS early life Stress, Childhood physical and sexual abuse, Patients with Combat-Related ( Vietnam combat veterans ), Patients Suffering from Headaches and without Therapy, combat-exposed U.S. Veterans). Sur la base de différentes cohortes allant de 23 à 60 ans, deux articles ont précisé le pourcentage de diminution du volume hippocampique droit et gauche (12% Left) et (8% Right).

Hippocampe et Alzheimer: 

D’après les articles étudiés [REV ASSOC MED BRAS 2020; 66(4):512-515, Structural Volume of Hippocampus and Alzheimer’s Disease, 10.1016/j.neuroscience.2015.08.033. inserm-01668526, http://www.cenir.org, REV ASSOC MED BRAS 2020; 66(4):512-515], observe une atrophie de l'hippocampe: l’imagerie de l’atrophie par IRM dans la maladie d’Alzheimer a montré selon une étude des articles étudiés une réduction du volume hippocampique, en remarquant une réduction de 10% suite aux troubles cognitifs légers, une baisse de 25% chez les sujets atteints de l’AD avec une forme légère et 40% chez les sujets ayant développés l’Alzheimer dans sa forme modérée.

II- Base de données:

BASE DE DONNEES IMAGES IRM CEREBRALES:
https://www.oasis-brains.org/#about

Cette base de données est une collecte d'images IRM cérébrales de 416 sujets agés entre 18 et 96 ans. Chaque sujet dispose de 3 ou 4 T1-weighted MRI scans.
Tous les sujtes sont des droitiers incluant des hommes et des femmes. Parmis ces sujets, on trouve 100 dont l'age est supérieur à 60 ans ont été diagnostiqués Alzheimer AD dans différents stades ( léger à modéré AD).

Sur la base de 457 sujets, nous avons identifié: 

      a- Les Nondemented (ND) avec un CDR (Clinical Dementia Rating)=0 ;
      b- Les demented (D) avec un CDR (Clinical Dementia Rating)=0.5, 1 et 2 (CDR; 0= nondemented; 0.5 – very mild dementia; 1 = mild dementia; 2 = moderate dementia) ;
      c- Les sujets sans CDR mentionné;
      e- Les sujets avec un (CDR >0) dont l'âge > 60 ans: atteints d’Alzheimer;
      f-  Les sujets avec un (CDR =0) dont l'âge > 60 ans: sujets contrôles;
      g- Le nombre d’hommes et de femmes ayant plus de 60 ans, avec un CDR=0 et CDR>0;
      h- Pour chaque catégorie, nous avons pris des tranches d'âges de 10 ans, pour pouvoir attribuer le même nombre d’hommes et de femmes des deux côtés pour chaque tranche.

Selon notre base de données, la coupe coronale est jugée la meilleure faisant apparaitre l'hippocampe, de ce faire on a décidé de segmenter que les coupes coronales:

77 hommes : 39 demented et 38 non-demented.

156 femmes : 62 demented et 94 non-demented.

III- Techniques de ségmentation:
1) Segmentation manuelle:
   
( M. Chupin et al  IRBM 32 (2011) 19–26  : Segmentation ciblée d’images IRM et maladie d’Alzheimer)

La segmentation manuelle sur les acquisitions pondérées en T1 est considérée comme la méthode la plus fiable pour obtenir un volume hippocampique, néanmoins la cette dernière technique continue de poser un certain nombre de problèmes pour son utilisation en routine : 
elle est très longue, demande une formation approfondie.
caractérisée par une variabilité intra- et inter-observateur qui demeure importante (erreur en volume entre 5 et 10 %, par exemple).
Tandis  que la segmentation entièrement automatique est l'objectif extrême. Elle a de nombreux avantages, en plus d'être indépendante de l'opérateur elle est rapide et reproductible.

2) Segmentation automatique:
   
La segmentation automatique permettrait de répondre à de nombreux problèmes rencontrés auparavant. Elle deviendrait alors un marqueur diagnostique utile pour la MA. Cependant, les limites macroscopiques de l’hippocampe sont incomplètes sur les acquisitions IRM à 1,5T ou 3T. Il devient alors nécessaire d’introduire de l’information a priori afin d’assurer une segmentation fiable et robuste. Cette information a priori peut provenir de plusieurs origines : sujet modèle, base d’apprentissage, atlas probabiliste ou connaissances anatomiques.

Nous allons réaliser la segmentation sur réseaux de neurones. Il existe plusieurs architectures de réseaux de neurones ( AlexNet , Resnet, GoogleNet, …) mais celle dédiée pour le traitement d’images médicales est U-NET qui a été  développée en 2015 par Olaf Ronneberger, depuis il y’a eu une apparition de ses variantes telles que : 2D U-NET, 3D U-NET, Cascad U-NET.

2-1- La segmentation par Deep Learning en utilisant l’architecture de Réseau de Neurones Convolutif

Dans le cadre de ce projet, nous nous sommes concentrés sur la segmentation automatisée des images IRM de l’hippocampe, en utilisant des réseaux de neurones convolutifs (CNN) basés sur des approches de Deep Learning. L'objectif principal de cette démarche est d'améliorer la précision et l'efficacité de la segmentation, tout en réduisant la dépendance à des méthodes manuelles sujettes à des variations inter-observateurs.

Le Deep Learning, en particulier les CNN, a émergé comme une solution puissante pour la segmentation d'organes dans des images médicales en raison de sa capacité à apprendre des caractéristiques complexes et hiérarchiques à partir de données. En adaptant ces techniques à la segmentation d’hippocampe en IRM, nous visons à fournir des outils précis, rapides et reproductibles pour l'analyse clinique.
Avant d’immerger dans l’implémentation de notre CNN, nous allons mettre la lumière sur la méthode de segmentation réalisée par la méthode de segmentation U-Net. 
II.2 .1 Architecture U-Net

Les réseaux de neurones convolutifs désignent une sous-catégorie de réseaux de neurones. Cependant, les CNN sont spécialement conçus pour traiter des images en entrée. Leur architecture est alors plus spécifique : elle est composée de deux blocs principaux [32] :
•	Le premier bloc : fait la particularité de ce type de réseaux de neurones, puisqu'il fonctionne comme un extracteur de features. Pour cela, en appliquant des opérations de filtrage par convolution. La première couche filtre l'image avec plusieurs noyaux de convolution, et renvoie des "feature-maps", qui sont ensuite normalisées (avec une fonction d'activation) et/ou redimensionnées. Ce procédé peut être réitéré plusieurs fois : on filtre les features maps obtenues avec de nouveaux noyaux, ce qui nous donne de nouvelles features maps à normaliser et redimensionner, et qu'on peut filtrer à nouveau, et ainsi de suite. Finalement, les valeurs des dernières feature maps sont concaténées dans un vecteur. Ce vecteur définit la sortie du premier bloc, et l'entrée du second.
•	Le second bloc : Les valeurs du vecteur en entrée sont transformées (avec plusieurs combinaisons linéaires et fonctions d'activation) pour renvoyer un nouveau vecteur en sortie. Ce dernier vecteur contient autant d'éléments qu'il y a de classes : l'élément i représente la probabilité que l'image appartient à la classe i.
 
Figure 10 Architecture U-Net 
https://www.geeksforgeeks.org/u-net-architecture-explained/)

Il existe quatre types de couches pour un réseau de neurones convolutif : la couche de convolution, la couche de max pooling, la couche de correction ReLU et la couche fully-connected.

•	Couche convolution 
La convolution, d’un point de vue simpliste, est le fait d’appliquer un filtre mathématique à une image. D’un point de vue plus technique, il s’agit de faire glisser une matrice par-dessus une image, et pour chaque pixel, utiliser la somme de la multiplication de ce pixel par la valeur de la matrice. Cette technique nous permet de trouver des parties de l’image qui pourraient nous être intéressantes. Prenons la Figure ci-dessous à gauche comme exemple d’image et la Figure à droite comme exemple de matrice [33].
 
Figure 11 Convolution : plusieurs filtres

•	Couche ReLu – Unité linéaire rectifiée 
ReLu est une fonction qui doit être appliquée à chaque pixel d’une image après convolution, et remplace chaque valeur négative par un 0. Si cette fonction n’est pas appliquée, la fonction créée sera linéaire et le problème XOR persiste puisque dans la couche de convolution, aucune fonction d’activation n’est appliquée [33].


•	Couche de Max pooling

Dans cette étape nous effectuons un sous-échantillonnage pour réduire le nombre de paramètres en ne gardant que l’activation la plus significative [33].
 
Figure 13 Max pooling

La couche de Max pooling permet de réduire le nombre de paramètres et de calculs dans le réseau. Nous améliorons ainsi l'efficacité du réseau et nous évitons le sur-apprentissage.

•	Couche fully-connected 

La couche fully-connected constitue toujours la dernière couche d'un réseau de neurones. Ce type de couche reçoit un vecteur en entrée et produit un nouveau vecteur en sortie. Pour cela, elle applique une combinaison linéaire puis éventuellement une fonction d'activation aux valeurs reçues en entrée. La dernière couche fully-connected permet de classifier l'image en entrée du réseau : elle renvoie un vecteur de taille N, où N est le nombre de classes dans notre problème de classification d'images. Chaque élément du vecteur indique la probabilité pour l'image en entrée d'appartenir à une classe [33].

Notre choix de l’architecture U-Net repose sur sa capacité à combiner une extraction efficace des caractéristiques, une compréhension hiérarchique du contexte spatial et une adaptabilité aux contraintes de données et de ressources informatiques. Sachant que ces caractéristiques font du modèle U-Net un choix judicieux pour répondre aux défis spécifiques de la segmentation de nos images IRM de l’hippocampe, offrant ainsi une base solide pour des résultats précis et reproductibles, et c’est ce qu’on voulait analyser par la suite.


Pour ce faire, on va suivre les étapes suivantes:

a- Préparation des données: données (training, validation et test) des images IRM avec leur segmentation manuelle de l’hippocampe correspondante.

b- Choix de l’architecture du RN: U-Net.

c- Entraînement du réseau choisi : permettre au réseau d’ajuster ces paramètres (selon les données d’entraînement) afin de pouvoir prédire la segmentation de l’hippocampe.

d- Evaluation du réseau choisi: cette étape s’effectuera à l’aide des données d'évaluation, pour pouvoir évaluer ce réseau en mesurant sa précision (Dice coefficient, Hausdorff mean...).

Après la prépration des données, on a classifié les images selon 3 coupes (axiale, sagittale et coronale), afin de les segmentés manuellement et de les mettre en entrainement.

Nous mettons à la disposition du public, nos données déjà triées et classées selon le sexe et la démence (frames et masques (labels) en format png).




