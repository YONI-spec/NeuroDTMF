
% Paramètres globaux
fs = 8000; %frequence d'echantillonage
duration = 0.2; % temps(200ms)
temps = 0:1/fs:duration; 
n_samples = 1000; % nombres de fichiers par touche

% Boucle sur les touches (0 à 9, *, #)
for
%   Pour chaque touche :
%      Calculer f_low et f_high
%      Boucle de 1 à n_samples :
%         1. Générer le signal pur
%         2. Déterminer un SNR aléatoire (ex: entre -5 et 20 dB)
%         3. Ajouter le bruit blanc (AWGN)
%         4. Appliquer un filtre passe-bande (optionnel mais pro)
%         5. Sauvegarder dans le dossier correspondant