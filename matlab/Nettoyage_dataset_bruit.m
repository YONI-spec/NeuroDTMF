clc; clear; close all;

input_dir = "16000";
output_dir = "real_noises";

fs_target = 8000;
target_duration = 1.0;

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, "*.wav"));

% filtre global
d_filter = designfilt('bandpassiir', ...
    'FilterOrder', 6, ...
    'HalfPowerFrequency1', 100, ...
    'HalfPowerFrequency2', 3500, ...
    'SampleRate', fs_target);

for i = 1:length(files)

    filepath = fullfile(files(i).folder, files(i).name);
    [audio, fs] = audioread(filepath);

    %% MONO
    if size(audio,2) > 1
        audio = mean(audio,2);
    end

    audio = audio';

    %% RESAMPLE
    if fs ~= fs_target
        audio = resample(audio, fs_target, fs);
    end

    %% NORMALISATION
    audio = audio / (max(abs(audio)) + 1e-6);

    %% FILTRE
    audio = filtfilt(d_filter, audio);

    %% SUPPRESSION SILENCE ADAPTATIVE
    threshold = 0.05 * max(abs(audio));
    idx = find(abs(audio) > threshold);

    if isempty(idx)
        continue;
    end

    margin = round(0.05 * fs_target);
    start_idx = max(1, idx(1) - margin);
    end_idx = min(length(audio), idx(end) + margin);

    audio = audio(start_idx:end_idx);

    %% DURÉE FIXE
    target_len = round(target_duration * fs_target);

    if length(audio) > target_len
        start_idx = randi(length(audio) - target_len);
        audio = audio(start_idx:start_idx + target_len - 1);
    else
        padding = 0.001 * randn(1, target_len - length(audio));
        audio = [audio padding];
    end

    %% NORMALISATION RMS
    rms_target = 0.1;
    audio = audio * (rms_target / (sqrt(mean(audio.^2)) + 1e-6));

    %% SAVE
    out_name = sprintf("noise_%d.wav", i);
    audiowrite(fullfile(output_dir, out_name), audio, fs_target);

end

disp("Dataset bruit amélioré prêt !");