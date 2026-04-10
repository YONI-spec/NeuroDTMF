clc; clear; close all;
rng(42); % reproductibilité

%% PARAMÈTRES 
fs = 8000;
n_samples = 1000;
output_dir = "dataset_spectrogram";

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% dossier silence (classe séparée)
silence_dir = fullfile(output_dir, "silence");
if ~exist(silence_dir, 'dir')
    mkdir(silence_dir);
end

%% IMPORT PYTHON 
d = py.dtmf_utils.DTMF_number;
keys_py = py.list(d.keys());
keys = string(cell(keys_py));

%% BRUIT RÉEL
real_noise_dir = "real_noises";
use_real_noise = exist(real_noise_dir, 'dir');

if use_real_noise
    noise_files = dir(fullfile(real_noise_dir, "*.wav"));
end

%% FILTRE 
d_filter = designfilt('bandpassiir', ...
    'FilterOrder', 6, ...
    'HalfPowerFrequency1', 600, ...
    'HalfPowerFrequency2', 1700, ...
    'SampleRate', fs);

%% FONCTION DE PROTECTION NaN/Inf
% Remplace tout NaN/Inf par 0 et évite la division par zéro
sanitize = @(x) x .* isfinite(x);

%% MAPPING NOMS DE DOSSIERS SÛRS
% '*' et '#' sont interdits dans les noms de fichiers/dossiers Windows
safe_name = containers.Map({'*', '#'}, {'star', 'hash'});



%% BOUCLE PRINCIPAL
for i = 1:length(keys)

    key = char(keys(i));
    freqs = double(py.array.array('d', d{key}));
    
    f_low  = freqs(1);
    f_high = freqs(2);

    % Nom de dossier sûr (* → star, # → hash)
    if safe_name.isKey(key)
        folder_name = safe_name(key);
    else
        folder_name = key;
    end
    key_dir = fullfile(output_dir, folder_name);
    if ~exist(key_dir, 'dir')
        mkdir(key_dir);
    end

    for k = 1:n_samples

        %% 1. DURÉE
        duration = 0.1 + rand()*0.2;
        t = (0:1/fs:(duration - 1/fs))';

        %% 2. JITTER + PHASE
        f_low_var  = f_low  + randn()*2;
        f_high_var = f_high + randn()*2;

        signal = sin(2*pi*f_low_var*t  + 2*pi*rand()) + ...
                 sin(2*pi*f_high_var*t + 2*pi*rand());

        %% 3. AMPLITUDE
        signal = (0.5 + rand()) * signal;

        %% 4. SHIFT TEMPOREL
        shift  = randi([0, round(0.02*fs)]);
        signal = [zeros(shift,1); signal];
        signal = signal(1:length(t));

        %% 5. BRUIT
        if rand() < 0.5
            snr = -5 + 25*rand();
            % awgn peut produire des NaN si le signal est nul
            signal_power = mean(signal.^2);
            if signal_power > 1e-10
                signal = awgn(signal, snr, 'measured');
            else
                signal = signal + randn(size(signal)) * 0.01;
            end

        elseif use_real_noise && ~isempty(noise_files)

            idx = randi(length(noise_files));
            noise_file = fullfile(noise_files(idx).folder, noise_files(idx).name);

            [real_noise, fs_n] = audioread(noise_file);

            if fs_n ~= fs
                real_noise = resample(real_noise, fs, fs_n);
            end

            real_noise = real_noise(:,1);

            if length(real_noise) < length(signal)
                real_noise = repmat(real_noise, ...
                    ceil(length(signal)/length(real_noise)), 1);
            end
            real_noise = real_noise(1:length(signal));

            snr_real     = -5 + 25*rand();
            signal_power = mean(signal.^2);
            noise_power  = mean(real_noise.^2);

            % Evite la division par zéro si le bruit réel est silencieux
            if noise_power > 1e-10
                k_noise = sqrt(signal_power / (noise_power * 10^(snr_real/10)));
                signal  = signal + k_noise * real_noise;
            end
        end

        % --- Protection post-bruit ---
        signal = sanitize(signal);

        %% 6. BRUIT FIN
        signal = signal + 0.005 * randn(size(signal));

        %% 7. ECHO
        echo_delay = round(0.02 * fs);
        echo       = [zeros(echo_delay,1); signal(1:end-echo_delay)];
        signal     = signal + (0.1 + 0.2*rand()) * echo;

        % --- Protection post-écho ---
        signal = sanitize(signal);

        %% 8. CLIPPING CONTRÔLÉ
        peak = max(abs(signal));
        if peak > 1e-6
            signal = signal / peak;
        end

        %% 9. FILTRE
        % filtfilt exige un signal fini — garanti par sanitize ci-dessus
        signal = filtfilt(d_filter, signal);

        % --- Protection post-filtre (sécurité supplémentaire) ---
        signal = sanitize(signal);

        %% 10. NORMALISATION
        peak = max(abs(signal));
        if peak > 1e-6
            signal = signal / peak;
        end

        %% 11. SILENCE (classe séparée)
        is_silence = rand() < 0.1;

        if is_silence
            signal = zeros(size(signal));
        end

        %% 12. SPECTROGRAMME
        win     = 256;
        overlap = 200;
        nfft    = 256;

        [s,~,~] = spectrogram(signal, win, overlap, nfft, fs);

        spec = log1p(abs(s));

        spec_std = std(spec(:));
        if spec_std > 1e-6
            spec = (spec - mean(spec(:))) / spec_std;
        else
            spec = spec - mean(spec(:));
        end

        img = imresize(mat2gray(spec), [128 128]);

        %% 13. SAVE
        if is_silence
            filename = fullfile(silence_dir, sprintf('silence_%d.png', k));
        else
            filename = fullfile(key_dir, sprintf('%s_%d.png', folder_name, k));
        end

        imwrite(img, filename);

    end

    fprintf("Classe %s générée (dossier: %s)\n", key, folder_name);
end

disp("Dataset final généré !");