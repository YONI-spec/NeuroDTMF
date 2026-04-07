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

%% BOUCLE PRINCIPAL
for i = 1:length(keys)

    key = char(keys(i));
    freqs = double(py.array.array('d', d{key}));
    
    f_low = freqs(1);
    f_high = freqs(2);

    key_dir = fullfile(output_dir, key);
    if ~exist(key_dir, 'dir')
        mkdir(key_dir);
    end

    for k = 1:n_samples

        %% 1. DURÉE
        duration = 0.1 + rand()*0.2;
        t = (0:1/fs:(duration - 1/fs))';

        %% 2. JITTER + PHASE
        f_low_var = f_low + randn()*2;
        f_high_var = f_high + randn()*2;

        signal = sin(2*pi*f_low_var*t + 2*pi*rand()) + ...
                 sin(2*pi*f_high_var*t + 2*pi*rand());

        %% 3. AMPLITUDE
        signal = (0.5 + rand()) * signal;

        %% 4. SHIFT TEMPOREL
        shift = randi([0, round(0.02*fs)]);
        signal = [zeros(shift,1); signal];
        signal = signal(1:length(t));

        %% 5. BRUIT
        if rand() < 0.5
            snr = -5 + 25*rand();
            signal = awgn(signal, snr, 'measured');

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
                    ceil(length(signal)/length(real_noise)),1);
            end
            real_noise = real_noise(1:length(signal));

            snr_real = -5 + 25*rand();

            signal_power = mean(signal.^2);
            noise_power = mean(real_noise.^2);

            k_noise = sqrt(signal_power / (noise_power * 10^(snr_real/10)));

            signal = signal + k_noise * real_noise;
        end

        %% 6. BRUIT FIN
        signal = signal + 0.005 * randn(size(signal));

        %% 7. ECHO
        echo_delay = round(0.02 * fs);
        echo = [zeros(echo_delay,1); signal(1:end-echo_delay)];
        signal = signal + (0.1 + 0.2*rand()) * echo;

        %% 8. CLIPPING CONTRÔLÉ
        if max(abs(signal)) > 1
            signal = signal / max(abs(signal));
        end

        %% 9. FILTRE
        signal = filtfilt(d_filter, signal);

        %% 10. NORMALISATION
        signal = signal / (max(abs(signal)) + 1e-6);

        %% 11. SILENCE (classe séparée)
        is_silence = rand() < 0.1;

        if is_silence
            signal = zeros(size(signal));
        end

        %% 12. SPECTROGRAMME
        win = 256;
        overlap = 200;
        nfft = 256;

        [s,~,~] = spectrogram(signal, win, overlap, nfft, fs);

        spec = log1p(abs(s)); % amélioration

        spec = (spec - mean(spec(:))) / (std(spec(:)) + 1e-6);

        img = imresize(mat2gray(spec), [128 128]);

        %% 13. SAVE
        if is_silence
            filename = fullfile(silence_dir, sprintf('silence_%d.png', k));
        else
            filename = fullfile(key_dir, sprintf('%s_%d.png', key, k));
        end

        imwrite(img, filename);

    end

    fprintf("Classe %s générée\n", key);
end

disp("Dataset final généré !");