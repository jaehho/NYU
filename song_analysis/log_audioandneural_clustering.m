%%
% Load the CSV file into a table
%T = readtable('audioCh4_annotations.csv');
%T = readtable('audioCh4_annotations4.csv');
T = readtable('Z:\longlab\homes\bahlea01\Ephys\Ma24\BudgieMa24_NLC\DAS\audioCh3_annotations.csv');
% Extract the 'name' column as a cell array of strings
type = T.name;

% Extract the 'start_seconds' and 'stop_seconds' columns as numeric vectors
syllable_onsets = T.start_seconds;
syllable_offsets = T.stop_seconds;

durations = syllable_offsets - syllable_onsets;
%%
% Find indices of syllables between 0.04 and 0.4 seconds
%validIdx = durations >= 0.1 & durations <= 0.5;
validIdx = durations >= 0.05 & durations <= 1;
validIdx();

% Filter variables
type = type(validIdx);

syllable_onsets = syllable_onsets(validIdx);
syllable_offsets = syllable_offsets(validIdx);

% type(strcmp(type, 'Song_proposals')) = {'Song'};
% type(strcmp(type, 'Low_proposals')) = {'Low'};
% 
%%
%idx = [find(strcmp(type, 'Song'));find(strcmp(type, 'Low'))];
idx = [find(strcmp(type, 'Song'))];
type = type(idx);
syllable_onsets = syllable_onsets(idx);
syllable_offsets = syllable_offsets(idx);
durations = syllable_offsets - syllable_onsets;
%%

[audioData, samplerate] = audioread('Z:\longlab\homes\bahlea01\Ephys\Ma24\BudgieMa24_NLC\audioCh3.flac');
%%
% Example parameters
hop_length = floor(2 * samplerate / 1000);  % hop length in samples
%win_length = floor(10 * samplerate / 1000);% * 2);  % window length in samples
win_length = floor(5 * samplerate / 1000);% * 2);  % window length in samples
entropy_variance = [];
mean_pitch = [];
% Note: MATLAB’s spectrogram uses window length and overlap, not n_fft directly
n_fft = win_length;  % FFT length, usually same as window length or power of 2

% Number of mel bands (adjust as you want)
numMelBands = 150;

specs = {};  % to store regular spectrograms of each segment
norm_specs = {};
for cnt = 1:length(syllable_onsets)
    onset = syllable_onsets(cnt);
    offset = syllable_offsets(cnt);
    
    % Convert onset/offset from seconds to sample indices
    startSample = floor(onset * samplerate) + 1;
    endSample = floor(offset * samplerate);
    
    % Extract audio segment
    segment = audioData(startSample:endSample);
    
    % Compute STFT using spectrogram (returns linear-frequency spectrogram)
    window = hann(win_length, 'periodic');
    [S, F, T] = spectrogram(segment, window, win_length - hop_length, n_fft, samplerate);
    P = abs(S).^2;                                            % Power spectrum
    P = P ./ sum(P,1);                                        % Normalize each time frame
    H = -sum(P .* log2(P + eps), 1);                          % Entropy per time frame
    entropy_variance(end+1) = var(H);                                % Variance of entropy
    % Power spectrogram
    specPower = abs(S).^2;
    noise_floor = min(specPower, [], 2);  % absolute minimum per frequency band
   % noise_floor = median(specPower, 2);
    
    normSpec = specPower ./ noise_floor;
    
    % Log2 scaling
    specLog2 = log2(normSpec + eps);
    
    % Subtract bias (e.g., equivalent to Python's spec - 2)
    specLog2 = specLog2 - 2;
    
    % Threshold
    specLog2(specLog2 < 0) = 0;
    
    % Optionally, downsample frequency axis like Python’s [4:-20:2] slice
    % specLog2 = specLog2(4:2:end-20, :);
    
    % Store result
    tt = specPower>prctile(specPower, 80);
    specs{end+1} = real(log(specPower));%tt;%(5:end-15,:);
    %specs{end+1} = tt;
    norm_specs{end+1}=normSpec;
end

%%
%[audioData, samplerate] = audioread('audioCh4.flac');
load('Z:\longlab\homes\bahlea01\Ephys\Ma24\BudgieMa24_NLC\sua.mat')
binWidth = 0.002;  % 5 ms
numNeurons = length(sua);
numSyllables = length(syllable_onsets);

firingRates = cell(numSyllables,1);  % each cell will be [neurons x bins]

parfor s = 1:numSyllables
    t0 = syllable_onsets(s);
    t1 = syllable_offsets(s);
    edges = t0:binWidth:t1;
    binCount = length(edges) - 1;
    rates = zeros(numNeurons, binCount);
    
    for n = 1:numNeurons
        spikes = sua(n).spikeTimes;
        counts = histcounts(spikes, edges);
        rates(n, :) = counts / binWidth;  % convert to Hz
    end
    
    firingRates{s} = rates;
end
%%
numPairs = length(specs);
resampledFiringRates = cell(numPairs, 1);

for i = 1:numPairs
    [nNeurons, oldBins] = size(firingRates{i});
    newBins = size(specs{i}, 2);

    % Interpolate firing rates to match spectrogram bin count
    resampledFiringRates{i} = interp1(...
        linspace(0, 1, oldBins), ...
        firingRates{i}', ...
        linspace(0, 1, newBins), ...
        'linear', 0 ...
    )';  % Transpose back to [neurons x bins]
end

%%
% Assumes specs and firingRates are cell arrays of same length
numPairs = length(specs);
combined = cell(numPairs, 1);

for i = 1:numPairs
    combined{i} = [specs{i}>prctile(specs{i}, 80); resampledFiringRates{i}>100];
    %combined{i} = [specs{i}>prctile(specs{i}, 80)];%
    %combined{i} = [resampledFiringRates{i}>100];
end
%%
specs = combined;
%%
scaling_factor = 8;  % adjust to control amount of compression
numSpecs = numel(specs);
resizedSpecs = cell(size(specs));

for k = 1:numSpecs
    S = double(specs{k});  % convert to double for imresize
    [freq_bins, time_bins] = size(S);
    
    % Calculate new time dimension size based on log scaling
    new_time_bins = max(1, round(log(time_bins) * scaling_factor));
    
    % Resize spectrogram matrix along time axis only
    % Keep freq_bins the same, change time_bins to new_time_bins
    resizedSpecs{k} = imresize(S, [freq_bins, new_time_bins], 'bicubic');
    %zetian style
    specs{k}= resizedSpecs{k};%>prctile(resizedSpecs{k}, 80);
    %time style
    %specs{k}= real(log(resizedSpecs{k}));%;
end


%%
% scaling_factor = 8;
% % 
% % % Apply log resize to all spectrograms
 spec_rs = cell(size(specs));
% for i = 1:length(specs)
%     spec_rs{i} = log_resize_spec(specs{i}, scaling_factor);
% end

% Find max time length after resize
max_len = max(cellfun(@(x) size(x, 2), specs));

% Pad all to the max length
for i = 1:length(spec_rs)
    spec_rs{i} = pad_end_spec(specs{i}, max_len);
end

% Assuming spec_rs is a cell array of spectrograms, each size [freq x time]
numSpecs = length(spec_rs);
freqBands = size(spec_rs{1}, 1);
timeFrames = size(spec_rs{1}, 2);

% Flatten each spectrogram into a row vector
spec_flat = zeros(numSpecs, freqBands * timeFrames);
for i = 1:numSpecs
    spec_flat(i, :) = spec_rs{i}(:);%reshape(spec_rs{i}, 1, []);
end

spec_flat(isnan(spec_flat)) = 0;

%%
rng(10)
addpath(genpath(fullfile(getenv('USERPROFILE'), 'Documents', 'umap')));
% spec_flat: numSpecs x numFeatures (flattened spectrograms)

% Run UMAP (2D embedding)
[embedding, umap_params] = run_umap(spec_flat(), ...
    'n_components', 2, ...
    'min_dist', 0.2, ...
    'n_neighbors', 10, ...
    'verbose', 'text');
%%

figure;
%scatter(embedding(:,1), embedding(:,2), 10, cluster_ids, 'filled');
%scatter(embedding(:,1), embedding(:,2), 10, 'filled');
clr = lines(10);

gscatter(embedding(:,1), embedding(:,2), type,clr);  % if categorical or numeric
xlabel('UMAP 1');
ylabel('UMAP 2');
title('UMAP of Spectrograms Colored by Cluster ID');
%colormap(lines); colorbar;

%%
% Run HDBSCAN (MATLAB version is available here: https://github.com/scikit-learn-contrib/hdbscan)
% Or alternatively, use DBSCAN with similar parameters if HDBSCAN unavailable
% Example with DBSCAN for approximation:
epsilon = 0.2; % radius parameter - tune accordingly
minpts = 10; % min cluster size approx.
%hdbscan_labels = dbscan([embedding,(durations)], epsilon, minpts);
hdbscan_labels = dbscan([embedding], epsilon, minpts);
% hdbscan_labels contains cluster indices (0 means noise)
%%

%%
% Run HDBSCAN (MATLAB version is available here: https://github.com/scikit-learn-contrib/hdbscan)
% Or alternatively, use DBSCAN with similar parameters if HDBSCAN unavailable
% Example with DBSCAN for approximation:

% epsilon = 0.35; % radius parameter - tune accordingly
% minpts = 5; % min cluster size approx.
% 
% epsilon = 0.15; % radius parameter - tune accordingly
% minpts = 10; % min cluster size approx.
%max_radius = 1.8; % your threshold

epsilon = 0.15; % radius parameter - tune accordingly
minpts = 5; % min cluster size approx.

%hdbscan_labels = dbscan([embedding,(durations)], epsilon, minpts);
hdbscan_labels = dbscan([embedding,durations], epsilon, minpts);
% hdbscan_labels contains cluster indices (0 means noise)

data = [embedding,durations];
unique_clusters = unique(hdbscan_labels);
unique_clusters(unique_clusters == -1) = []; % remove noise label

    max_radius = 133.34; % your threshold

valid_clusters = hdbscan_labels;
i = 1;
for c = unique_clusters'
    cluster_points = data(hdbscan_labels == c, :);
    centroid = mean(cluster_points, 1);
    distances = sqrt(sum((cluster_points - centroid).^2, 2));
    cluster_radius(i) = max(distances); % or use prctile(distances, 95);
    
    if cluster_radius(i) >= max_radius
        valid_clusters(hdbscan_labels == c) = -1;
    end
    i = i+1;
end

hdbscan_labels = valid_clusters;
figure;
%scatter(embedding(:,1), embedding(:,2), 10, cluster_ids, 'filled');
%scatter(embedding(:,1), embedding(:,2), 10, 'filled');
clr = lines(10);

gscatter(embedding(:,1), embedding(:,2), hdbscan_labels,clr);  % if categorical or numeric
xlabel('UMAP 1');
ylabel('UMAP 2');
title('UMAP of Spectrograms Colored by Cluster ID');
%colormap(lines); colorbar;
%%
addpath(genpath('C:\Users\ADMIN\Documents\matplotlib'))
%colororder(tab10(length(categories_list)))
%hdbscan_labels = hdbscan_labels_neural;
type_cat = categorical(hdbscan_labels);

% Get unique category names
categories_list = categories(type_cat);

% Choose a colormap (e.g., lines, parula, distinguishable_colors from File Exchange, etc.)
%colors = lines(length(categories_list));  % or try 'parula', 'jet', etc.
colors = colorcube(length(categories_list));  % or try 'parula', 'jet', etc.
colors = tab20(length(categories_list));
figure;
hold on;

% Plot each category separately with its own color
for i = 2:length(categories_list)
    idx = type_cat == categories_list{i};
    sz(i) = sum(idx);
    szdur(i) = median(durations(idx));
    plot3(embedding(idx,1)+rand(size(embedding(idx,1)))/5, embedding(idx,2)+rand(size(embedding(idx,1)))/5, log(durations(idx)), '.', ...
          'Color', colors(i,:), 'DisplayName', char(categories_list{i}),'MarkerSize',5);
end

for i = 1
    idx = type_cat == categories_list{i};
     szdur(i) = median(durations(idx));
sz(i) = [];
    plot3(embedding(idx,1), embedding(idx,2), log(durations(idx)), '.', 'Color', [1,1,1]./1,'MarkerSize',4);
end

% for i = 59
%     idx = type_cat == categories_list{i};
%      szdur(i) = median(durations(idx));
% sz(i) = [];
%     plot3(embedding(idx,1), embedding(idx,2), log(durations(idx)), 'ko', ...
%           'Color', [1,1,1].*0,'MarkerSize',10);
% end

xlabel('t-SNE 1');
ylabel('t-SNE 2');
zlabel('Duration');
title('t-SNE of Flattened Spectrograms Colored by Type');
legend('off');
grid on;
axis off
title('')
view(2);  % Ensure 3D view is enabled
%%
idx = type_cat == categories_list{1};
rng(10)
addpath(genpath(fullfile(getenv('USERPROFILE'), 'Documents', 'umap')));
% spec_flat: numSpecs x numFeatures (flattened spectrograms)

% Run UMAP (2D embedding)
[embedding, umap_params] = run_umap(spec_flat(~idx,:), ...
    'n_components', 2, ...
    'min_dist', 0.1, ...
    'n_neighbors', 10, ...
    'verbose', 'text');
%%
figure;
%scatter(embedding(:,1), embedding(:,2), 10, cluster_ids, 'filled');
%scatter(embedding(:,1), embedding(:,2), 10, 'filled');
clr = lines(10);

gscatter(embedding(:,1), embedding(:,2), hdbscan_labels(~idx),clr);  % if categorical or numeric
xlabel('UMAP 1');
ylabel('UMAP 2');
title('UMAP of Spectrograms Colored by Cluster ID');
%colormap(lines); colorbar;
%%
idx = type_cat == categories_list{1};
type_cat = categorical(hdbscan_labels);
ddurations = durations(~idx);
% Get unique category names
categories_list = categories(type_cat(~idx));

% Choose a colormap (e.g., lines, parula, distinguishable_colors from File Exchange, etc.)
colors = lines(length(categories_list));  % or try 'parula', 'jet', etc.

figure;
hold on;

% Plot each category separately with its own color
for i = 2:length(categories_list)
    idx = type_cat == categories_list{i};
    %sz(i) = sum(idx);
    %szdur(i) = median(ddurations(idx));
    plot3(embedding(idx,1), embedding(idx,2), log(ddurations(idx)), '.', ...
          'Color', colors(i,:), 'DisplayName', char(categories_list{i}),'MarkerSize',3);
end

for i = 3%:length(categories_list)
    idx = type_cat == categories_list{i};
    %sz(i) = sum(idx);
    %szdur(i) = median(ddurations(idx));
    plot3(embedding(idx,1), embedding(idx,2), log(ddurations(idx)), '.', ...
          'Color', colors(i,:), 'DisplayName', char(categories_list{i}),'MarkerSize',10);
end


xlabel('t-SNE 1');
ylabel('t-SNE 2');
zlabel('Duration');
title('t-SNE of Flattened Spectrograms Colored by Type');
legend('off');
grid on;
view(3);  % Ensure 3D view is enabled
%%
% Prepare data (excluding first element as in original code)
[y,ri] = sort(sz(2:end), 'descend');
x = 1:length(y);
figure,
loglog(x, y, 'b'); hold on;
xlabel('Index'); ylabel('Value'); title('Log-Log Plot');
r = corr(log(x(:)), log(y(:)), 'Type', 'Pearson');
r^2
%%
% Sort and exclude the first point if needed (often 0 or outlier)
data = sort(sz(2:end), 'descend');
x = (1:length(data))'; % rank or index
y = data;

% Convert to log-log
logx = log10(x);
logy = log10(y);

% Fit linear model in log-log space
coeffs = polyfit(logx, logy, 1);
alpha = -coeffs(1); % Power-law exponent

% Generate fit line
yfit = 10.^(polyval(coeffs, logx));

% Plot
figure;
loglog(x, y, 'b', 'LineWidth', 1.5); hold on;
loglog(x, yfit, 'r--', 'LineWidth', 2);
xlabel('Rank'); ylabel('Value');
title(sprintf('Power-law fit: exponent \\alpha = %.2f', alpha));
legend('Data', 'Power-law fit');
%%
% Assume `sz` contains sizes, sorted in descending order
data = sort(sz(2:end), 'descend'); % skip first if needed
ranks = (1:length(data))'; % rank vector

% Convert to log-log
logRanks = log(ranks);
logData = log(data);

% Linear fit
coeffs = polyfit(logRanks, logData, 1);
s = -coeffs(1); % Zipf exponent

% Predicted Zipf model
fitLine = exp(polyval(coeffs, logRanks));

% Plot in log-log
figure;
loglog(ranks, data, 'b-', 'LineWidth', 1.5); hold on;
loglog(ranks, fitLine, 'r--', 'LineWidth', 2);
xlabel('Rank');
ylabel('Size / Frequency');
title(sprintf('Zipf Fit: Exponent s = %.2f', s));
legend('Data', 'Zipf Fit');

%%
% Assume `sz` contains sizes, sorted in descending order
data = sort(sz(2:end), 'descend'); % Skip first if needed
ranks = (1:length(data))'; % Rank vector (column)

% Define Zipf-Mandelbrot model: f(r) = A / (r + b)^s
zipf_mandel = @(params, r) params(1) ./ (r + params(2)).^params(3);  % Returns column

% Initial parameter guess: [A, b, s]
A0 = data(1);
b0 = 1;
s0 = 1.0;
initParams = [A0, b0, s0];

% Make sure data is a column vector
data = data(:);
ranks = ranks(:);

% Fit using nonlinear least squares
opts = optimset('Display','off');
fitParams = lsqcurvefit(zipf_mandel, initParams, ranks, data, [], [], opts);

% Extract fitted parameters
A = fitParams(1);
b = fitParams(2);
s = fitParams(3);
fitLine = zipf_mandel(fitParams, ranks);

% Plot in log-log scale
figure;
loglog(ranks, data, 'k-', 'LineWidth', 1.5); hold on;
loglog(ranks, fitLine, 'r--', 'LineWidth', 2);
xlabel('Rank');
ylabel('Size / Frequency');
title(sprintf('Zipf-Mandelbrot Fit: A=%.1f, b=%.2f, s=%.2f', A, b, s));
legend('Data', 'Fit');
grid on;

%% Split the data in half to seperately estimate rank and freuqncy 
% Ensure reproducibility
rng(1);

% Total number of tokens (assumes sz is a list of token types by count)
syllableCounts = sz(:);  % Ensure column
N = sum(syllableCounts);  % Total token count

% Create a pool of tokens by type index
tokenPool = arrayfun(@(i) repmat(i, syllableCounts(i), 1), 1:length(syllableCounts), 'UniformOutput', false);
tokenPool = vertcat(tokenPool{:});

% Shuffle tokens
shuffled = tokenPool(randperm(N));

% Split into two halves
halfPoint = floor(N / 2);
firstHalf = shuffled(1:halfPoint);
secondHalf = shuffled(halfPoint+1:end);

% Count frequencies in each half
numTypes = length(syllableCounts);
freq1 = accumarray(firstHalf, 1, [numTypes, 1]);
freq2 = accumarray(secondHalf, 1, [numTypes, 1]);

% Use freq1 to get frequencies (Y)
% Use freq2 to rank types
[sortedFreq2, idx2] = sort(freq2, 'descend');
validIdx = sortedFreq2 > 0;  % Remove zero-frequency entries
ranks = (1:sum(validIdx))';
data = freq1(idx2(validIdx));

% Remove zero entries in freq1 to avoid log(0)
nonZero = data > 0;
data = data(nonZero);
ranks = ranks(nonZero);

% Fit Zipf-Mandelbrot: f(r) = A / (r + b)^s
zipf_mandel = @(params, r) params(1) ./ (r + params(2)).^params(3);
initParams = [max(data), 1, 1];
opts = optimset('Display','off');

fitParams = lsqcurvefit(zipf_mandel, initParams, ranks, data, [], [], opts);

% Extract fitted parameters
A = fitParams(1);
b = fitParams(2);
s = fitParams(3);
fitLine = zipf_mandel(fitParams, ranks);

coeffs = polyfit(log(ranks), log(data), 1);
ss = -coeffs(1); % Zipf exponent

% Predicted Zipf model
fitLine2 = exp(polyval(coeffs, log(ranks)));

% Plot
figure;
loglog(ranks, data, 'ko-', 'LineWidth', 1.5); hold on;
loglog(ranks, fitLine, 'r--', 'LineWidth', 2);
%loglog(ranks, fitLine2, 'b--', 'LineWidth', 2);
xlabel('Rank (from 2nd half)');
ylabel('Frequency (from 1st half)');
title(sprintf('Cross-validated Zipf-Mandelbrot Fit: A=%.1f, b=%.2f, s=%.2f', A, b, s));
legend('Data', 'Fit');
%grid on;
set(gca, 'TickDir', 'out')
%%
% Prepare data (excluding first element as in original code)
y = sort(sz(2:end), 'descend')';
x = 1:length(y);

% Linear plot with exponential fit
figure;
plot(x, y, 'b'); hold on;
xlabel('Index'); ylabel('Value'); title('Linear Scale');

% Fit: y = a * exp(-b * x)
ft = fittype('a * exp(-b * x)', 'independent', 'x', 'coefficients', {'a', 'b'});
fitResult = fit(x', y, ft);
plot(x, fitResult.a * exp(-fitResult.b * x), 'r--', 'LineWidth', 2);
legend('Data', 'Exponential Fit');

% Log plot with log-linear fit
figure;
semilogy(x, y, 'b'); hold on;
xlabel('Index'); ylabel('Value (log scale)'); title('Log Scale');

% Fit linear model to log(y): log(y) = log(a) - b * x  --> y = a * exp(-b * x)
logY = log(y);
p = polyfit(x, logY, 1);
logFit = exp(p(2)) * exp(p(1) * x);
semilogy(x, logFit, 'r--', 'LineWidth', 2);
legend('Data', 'Log-Linear Fit');
%%
figure('Position',[100 100 1200 600])

% Manual syllable labels
subplot(1,2,1)
gscatter(embedding(:,1), embedding(:,2))
xlabel('UMAP 1')
ylabel('UMAP 2')
title('Manual syllable labels')
alpha(0.2) % set transparency

% Create custom colormap with gray for cluster 0
baseCmap = lines(max(hdbscan_labels)+1); % or any colormap you like
grayColor = [0.7 0.7 0.7];
if any(hdbscan_labels == 0)
    baseCmap(1,:) = grayColor; % assuming label 0 maps to first color
end

% Plot HDBSCAN clusters
subplot(1,2,2)
% To handle cluster 0 (outliers) coloring with gray:
% Replace 0 with 1 for colormap indexing, but plot with gray for those points.
labels_plot = hdbscan_labels;
labels_plot(labels_plot==0) = 1; 

scatter(embedding(:,1), embedding(:,2), 8, baseCmap(labels_plot,:), 'filled')
xlabel('UMAP 1')
ylabel('UMAP 2')
title('Unsupervised syllable labels')
alpha(0.2) % transparency

% Optional: add legend for clusters, tweak colormap as needed

%%
% Logarithmic resize function (time axis only)
function spec_resized = log_resize_spec(spec, scaling_factor)
    [numBands, numFrames] = size(spec);
    newLen = max(1, floor(log(double(numFrames)) * scaling_factor)); % avoid log(0)
    
    % Original time points (linear)
    t_lin = linspace(1, numFrames, numFrames);
    % New time points (linear spaced, will interpolate on original linear scale)
    t_new = linspace(1, numFrames, newLen);
    
    % Interpolate each freq band along time axis
    spec_resized = zeros(numBands, newLen);
    for b = 1:numBands
        spec_resized(b, :) = interp1(t_lin, spec(b, :), t_new, 'linear', 0);
    end
end

% Pad symmetrically on time axis to target length
function spec_padded = pad_spec(spec, pad_length)
    [numBands, currLength] = size(spec);
    excess = pad_length - currLength;
    pad_left = floor(excess / 2);
    pad_right = ceil(excess / 2);
    
    spec_padded = padarray(spec, [0 pad_left], 0, 'pre');
    spec_padded = padarray(spec_padded, [0 pad_right], 0, 'post');
end

function spec_padded = pad_end_spec(spec, pad_length)
    [numBands, currLength] = size(spec);
    excess = pad_length - currLength;

    % Only pad if padding is needed
    if excess > 0
        spec_padded = padarray(spec, [0 excess], 0, 'post'); % Pad at the end (right)
    else
        spec_padded = spec(:, 1:pad_length); % Trim if too long
    end
end