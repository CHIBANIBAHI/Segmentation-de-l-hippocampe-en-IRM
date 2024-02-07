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

Figure 1 Architecture U-Net 
![img.jpg](https://github.com/CHIBANIBAHI/Segmentation-de-l-hippocampe-en-IRM/blob/main/img.jpg)


Il existe quatre types de couches pour un réseau de neurones convolutif : la couche de convolution, la couche de max pooling, la couche de correction ReLU et la couche fully-connected.

•	Couche convolution 
La convolution, d’un point de vue simpliste, est le fait d’appliquer un filtre mathématique à une image. D’un point de vue plus technique, il s’agit de faire glisser une matrice par-dessus une image, et pour chaque pixel, utiliser la somme de la multiplication de ce pixel par la valeur de la matrice. Cette technique nous permet de trouver des parties de l’image qui pourraient nous être intéressantes. Prenons la Figure ci-dessous à gauche comme exemple d’image et la Figure à droite comme exemple de matrice.

•	Couche ReLu – Unité linéaire rectifiée 
ReLu est une fonction qui doit être appliquée à chaque pixel d’une image après convolution, et remplace chaque valeur négative par un 0. Si cette fonction n’est pas appliquée, la fonction créée sera linéaire et le problème XOR persiste puisque dans la couche de convolution, aucune fonction d’activation n’est appliquée.

•	Couche de Max pooling

Dans cette étape nous effectuons un sous-échantillonnage pour réduire le nombre de paramètres en ne gardant que l’activation la plus significative.

La couche de Max pooling permet de réduire le nombre de paramètres et de calculs dans le réseau. Nous améliorons ainsi l'efficacité du réseau et nous évitons le sur-apprentissage.

•	Couche fully-connected 

La couche fully-connected constitue toujours la dernière couche d'un réseau de neurones. Ce type de couche reçoit un vecteur en entrée et produit un nouveau vecteur en sortie. Pour cela, elle applique une combinaison linéaire puis éventuellement une fonction d'activation aux valeurs reçues en entrée. La dernière couche fully-connected permet de classifier l'image en entrée du réseau : elle renvoie un vecteur de taille N, où N est le nombre de classes dans notre problème de classification d'images. Chaque élément du vecteur indique la probabilité pour l'image en entrée d'appartenir à une classe.

Notre choix de l’architecture U-Net repose sur sa capacité à combiner une extraction efficace des caractéristiques, une compréhension hiérarchique du contexte spatial et une adaptabilité aux contraintes de données et de ressources informatiques. Sachant que ces caractéristiques font du modèle U-Net un choix judicieux pour répondre aux défis spécifiques de la segmentation de nos images IRM de l’hippocampe, offrant ainsi une base solide pour des résultats précis et reproductibles, et c’est ce qu’on voulait analyser par la suite.

Pour ce faire, on va suivre les étapes suivantes:

a- Préparation des données: données (training, validation et test) des images IRM avec leur segmentation manuelle de l’hippocampe correspondante.

b- Choix de l’architecture du RN: U-Net.

c- Entraînement du réseau choisi : permettre au réseau d’ajuster ces paramètres (selon les données d’entraînement) afin de pouvoir prédire la segmentation de l’hippocampe.

d- Evaluation du réseau choisi: cette étape s’effectuera à l’aide des données d'évaluation, pour pouvoir évaluer ce réseau en mesurant sa précision (Dice coefficient, Hausdorff mean...).

Après la prépration des données, on a classifié les images selon 3 coupes (axiale, sagittale et coronale), afin de les segmentés manuellement et de les mettre en entrainement.

Nous mettons à la disposition du public, nos données déjà triées et classées selon le sexe et la démence (frames et masques (labels) en format png).

3) La segmentation manuelle: vérité térrain

![seg1.png](https://github.com/CHIBANIBAHI/Segmentation-de-l-hippocampe-en-IRM/blob/main/seg1.png)

Cette étape consiste à segmenter manuellement les différentes images des sujets de la base de données afin de pouvoir entrainer notre réseau par des masques de la vérité terrain.
Afin de procéder à cette opération essentielle et demandant une très grande concentration et pour pouvoir bien délimiter l’hippocampe dans les différentes coupes, nous avons été guidée par la neurologue Dr Anna SONTHEIMER qui nous a orienté vers la bonne méthode de segmentation, ainsi après des réunions de vérification et d’après les coupes présentes dans notre base de donnée, nous avons décidé de segmenter que la coupe coronale, et ce pour sa meilleure présentation de l’hippocampe et dans le sens où ce dernier apparait clairement sur la coupe mentionnée.
Pour pouvoir faire cette segmentation manuelle, nous avons fait plusieurs recherches de logiciels (napari, volbrain et autres), mais nous avons opté pour le logiciel MITK développé par l’équipe de l’institut Pascal de Clermont-Ferrand.

Ce logiciel permet la segmentation experte de 3 coupes d’images, tout en donnant par la suite un volume des 3 coupes.

Dans notre projet nous avons segmenté une seule coupe en utilisant ce logiciel.

4) Calcul de la surface hippocampique
![surface.png](https://github.com/CHIBANIBAHI/Segmentation-de-l-hippocampe-en-IRM/blob/main/surface.png)

Cette étape consiste à calculer la surface hippocampique des images des masques obtenus par segmentation manuelle et de la comparer avec celle de la segmentation automatique dont le but de comparer la surface des sujets sains et déments, et ce afin d’avoir une idée sur le pourcentage de diminution de la surface hippocampique chez les sujets atteints d’Alzheimer pour valider nos recherches bibliographiques.
Dans notre projet et comme nous avons déjà cité ci-dessus que nous avons juste la segmentation des coupes coronales, nous nous sommes basés sur la surface de l’hippocampe et ce en utilisant le logiciel MITK.

Le logiciel MITK nous permets d’utiliser le volet « Measurement » qui donne la possibilité de mesurer la surface (Area) de la partie délimitée de l’hippocampe, et ce en utilisant un polygone.
Après avoir mesurer la surface de l’hippocampe segmentée de toutes les coupes coronales de nos images, nous avons pu remplir notre tableau qui va nous servir de base pour le calcul de la moyenne des surfaces des images et de l’écart-type par sexe et par démence.

5) Résultats:

![tab.PNG](https://github.com/CHIBANIBAHI/Segmentation-de-l-hippocampe-en-IRM/blob/main/tab.PNG)

Pendant cette étape, nous avons procéder à la classification des sujets de la base de données comme suit :
•	Groupe homme demented : 39 sujets
•	Groupe homme non-demented : 38 sujets
•	Groupe femme demented : 62 sujets
•	Groupe femme non-demented : 94 sujets
Pour chaque catégorie citée ci-dessus, nous avons calculé la moyenne de la surface et l’écart type correspondant de l’hippocampe (droit et gauche). Résultats présentés dans le tableau suivant :

6) Interprétations

D’après ces résultats, il nous est apparu que dans le cas d’une atteinte d’Alzheimer il y a une diminution de la surface de l’hippocampe en se comparant aux sujets sains.
Comparaison entre hommes atteints d'Alzheimer et hommes non-demented

•	Pour les hommes atteints d'Alzheimer, la moyenne de la surface de l'hippocampe droit (107,021 mm2) et gauche (107,198 mm2) est inférieure par rapport aux hommes non-demented (119,133 mm2 pour le droit et 128,324 mm2 pour le gauche). Cela suggère une réduction significative de la surface de l'hippocampe chez les hommes atteints de la maladie d'Alzheimer.

Comparaison entre femmes atteintes d'Alzheimer et femmes non-demented
•	Chez les femmes atteintes d'Alzheimer, la moyenne de la surface de l'hippocampe droit (99,141 mm2) et gauche (105,985 mm2) est également inférieure par rapport aux femmes non-demented (117,819 mm2 pour le droit et 125,900 mm2 pour le gauche). Cette différence suggère une réduction de la surface de l'hippocampe chez les femmes atteintes de la maladie d'Alzheimer.

Comparaison entre hommes et femmes non-demented
•	Il semble y avoir des différences de base entre hommes et femmes non-demented. Les hommes semblent avoir une moyenne plus élevée de la surface de l'hippocampe par rapport aux femmes, tant pour le côté droit que le côté gauche. Ces différences peuvent être liées aux variations normales de la morphologie cérébrale entre les sexes.
Variabilité au sein des groupes 
•	L'écart type mesure la dispersion des valeurs autour de la moyenne. Pour tous les groupes, l'écart type est relativement modéré, indiquant une certaine cohérence dans les données. Cependant, la variabilité semble être plus importante chez les femmes atteintes d'Alzheimer, en particulier pour le côté droit (sachant que le nombre des sujets n’est pas similaire entre deux sexes).

Asymétrie entre les hémisphères 
•	Pour tous les groupes, la moyenne de la surface de l'hippocampe droit est généralement inférieure à celle du côté gauche. Cela peut refléter des asymétries normales dans la structure cérébrale, bien que cela puisse également être influencé par d'autres facteurs.
En résumé, ces résultats suggèrent des différences significatives dans la surface de l'hippocampe entre les groupes étudiés, avec une réduction observée chez les individus atteints de la maladie d'Alzheimer par rapport à ceux en bonne santé, et des variations entre hommes et femmes. L'écart type fournit également des indications sur la dispersion des données au sein de chaque groupe.

Conclusion et perspectives

En conclusion, la segmentation automatique s’est avérée robuste dans la délimitation des régions hippocampique, les résultats de cette étude soulignent l'importance de la segmentation d'images médicales dans l'analyse du volume de l'hippocampe, une méthode essentielle pour le diagnostic de pathologies ciblées, notamment les altérations du volume hippocampique associées à des troubles cérébraux tels que la maladie d'Alzheimer, ou à l’ESPT.
Au cours de ce projet, nous avons développé un algorithme de segmentation automatique basé sur l'architecture U-Net. Avant cette étape, une segmentation manuelle a été réalisée pour établir une référence. Après avoir obtenu les contours précis de l'hippocampe, nous avons calculé la surface hippocampique, mettant en lumière une diminution significative chez les individus atteints de la maladie d'Alzheimer, en accord avec les observations de la littérature.
Ces résultats suggèrent que l'analyse automatisée de la morphologie cérébrale peut non seulement confirmer les tendances décrites dans la littérature, telles que la réduction du volume hippocampique, mais également offrir une approche plus rapide et potentiellement plus précise pour le diagnostic.
En intégrant une segmentation manuelle préalable, notre étude renforce la robustesse de l'approche automatisée. Ces avancées méthodologiques ouvrent des perspectives prometteuses pour une utilisation clinique plus généralisée de l'imagerie médicale dans le diagnostic précoce et la compréhension des maladies cérébrales, offrant ainsi des opportunités pour des interventions plus ciblées et efficaces.
De la contrainte du temps, notre projet s'est concentré sur le développement d'un algorithme de segmentation automatique de l'hippocampe, avec pour objectif initial le calcul du volume hippocampique. Bien que la comparaison entre les volumes des sujets atteints de pathologies cognitives et ceux des sujets sains n'ait pas été entièrement réalisée dans le cadre de ce projet, cette étape représente une perspective majeure pour une poursuite de nos travaux.
Parallèlement, nous envisageons d'explorer des techniques avancées d'apprentissage automatique pour améliorer la précision de la segmentation et optimiser la fiabilité des mesures de volume, en superposons les trois coupes d’images IRM, ainsi de calculer le volume de la segmentation automatique et de le comparer à celui de la segmentation manuelle afin de valider la littérature.
Finalement, bien que notre projet ait rencontré des contraintes temporelles, il a jeté les bases solides d'une recherche prometteuse dans le domaine de la segmentation automatique de l'hippocampe. Les perspectives à venir visent à élargir nos analyses, améliorer la précision de nos résultats et renforcer l'applicabilité clinique de notre approche.





