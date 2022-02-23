# IFT6759 Deep learning:  ANimal Deep learning Identification ANDI
This repo is for the project of animal classification using deep learning. 

#Team Members
- Jonathan Beaulieu-Emond jonathan.beaulieu-emond@umontreal.ca
- Selim Gilon selim.gilon@umontreal.ca
- Yassine Kassis yassine.kassis.1@umontreal.ca
- Jean-francois Girard-Baril jean-francois.girard-baril@umontreal.ca

Project Gantt chart: https://docs.google.com/spreadsheets/d/1dAW6vDA6k7e2ML3V-6WNNC8MwGt3SMAw1tNjU0YOjMg/edit?usp=sharing

#Description du projet

Notre projet se basera sur la classification d’animaux à l’aide de modèles pré-entraînés. Le but sera d'entraîner le modèle avec un nombre limité d’entrées afin de tester la capacité du modèle de s’adapter à des photos prises dans d’autres environnements. Pour ce faire, nous utiliserons VGG19, yoloV5, InceptionV4, et/ou faster R-CNN.

Ces modèles seront ré-entrainé avec un subset des données (environ 15%) . Le but sera alors de porter ce modèle, entrainé sur seulement quelques localisations, vers les autres.
Nous essayerons également d’utiliser quelques techniques afin d’aider notre modèle à généraliser, dont lla technique de domain adaptation. 


#Étapes importantes

##Milestone 1 (25 février): Revue de littérature , avoir trouvé une architecture de réseau à développer ET trouver un modèle pré-existant pour comparer nos futurs résultats

##Milestone 2 (4 mars): Prétraitement des données, avoir un “dataloader”/dataset prêt.


##Milestone 3 (11 mars) : Résultats préliminaire

##Milestone 4 (16 mars) : Amélioration des modèles

##Milestone 5 (18 mars) : Résultats préliminaires (2 pager)

##Milestone 6(23 mars) : Résultats finaux modèle classification

##Milestone 7 (1er avril) : Résultats finaux modèle


##Milestone 8(13 avril) : présentation en classe

##Milestone 9: Écriture du rapport

##Milestone 10 : 29 avril : Rapport final





#Distribution des tâches

Nous utiliserons principalement la méthode de gestion Kanban implémentée avec github project pour la gestion de la distribution du travail.

Pour permettre ce type de travail décentralisé, comet.ml sera également utilisé afin de permettre d’enregistrer nos expérience effectué sur nos différents modèles lors de l'entraînement, permettant à tous les membres d’avoir accès aux résultats.

De plus, il fut décidé qu’une rencontre bi-hebdomadaire, les mercredi et vendredi, serait organisée afin de travailler sur le projet en équipe pour donner l’opportunité à chaque membre de travailler et réviser chacune des étapes du projet.
