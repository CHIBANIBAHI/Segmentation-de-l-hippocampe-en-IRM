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
      On a obtenu le nombre de sujets jugés utiles pour notre segmentation: 166 pour chaque coupe.

Problématique: Ce nombre nous apparaît insuffisant !

Solution proposée: étaler la tranche d'âge de sélection!  Possibilité d'ajouter autres bases de données..

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

Pour ce faire, on va suivre les étapes suivantes:

a- Préparation des données: données (training, validation (Question: ces données doivent être prises d’une autre base de données ou bien celles du test?) et test) des images IRM avec leur segmentation manuelle de l’hippocampe ( et ou sans amygdale)  correspondante.
b- Choix de l’architecture du RN: U-Net 3D (Keras) GitHub - davidiommi/3D-U-net-Keras: 3D-Unet: patched based Keras implementation for medical images segmentation
c- Entraînement du réseau choisi : permettre au réseau d’ajuster ces paramètres (selon les données d’entraînement) afin de pouvoir prédire la segmentation de l’hippocampe ( et ou sans amygdale).
d- Evaluation du réseau choisi: cette étape s’effectuera à l’aide des données d'évaluation, pour pouvoir évaluer ce réseau en mesurant sa précision.

On est dans la phase de préparation de la base de données, en triant les images selon 3 coupes (axiale, sagittale et coronale), afin de les segmentés manuellement et de les mettre en entrainement.


