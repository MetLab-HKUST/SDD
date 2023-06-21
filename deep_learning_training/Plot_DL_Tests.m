%% load data 
obsMax = ncread('rxnet_rec_max_only_test.nc', 'obsMax');
obsMax = obsMax';

recMax = ncread('rxnet_rec_max_only_test.nc', 'recMax');
recMax = recMax';

% sort data so we only plot top 1000 cases
[obsSort, ~] = sort(obsMax, 'descend');

lowBound = obsSort(2000);
disp(lowBound)

obsPlot = obsMax(obsMax>=lowBound);
recPlot = recMax(obsMax>=lowBound);

%% plot all
tiledlayout(1, 2, "TileSpacing", "compact")

nexttile 
s1 = scatter(obsMax, recMax, (30+obsMax).^2/3600, 'blue', 'filled');
%s1.AlphaData = (0.2+obsMax / max(obsMax)).^(1/2);
%s1.MarkerFaceAlpha = 'flat';
ylim([1e0 240])
xlim([1e0 240])
xticks([1.0 10.0 100])
yticks([1.0 10.0 100])
pbaspect([1 1 1])
grid on
box on
xlabel('observation (mm/day)')
ylabel('DL prediction (mm/day)')
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')
ttl = title('a)');
ttl.Units = 'Normalize'; 
ttl.Position(1) = 0; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 

% %% plot top 1000
% nexttile 
% s2 = scatter(obsPlot, recPlot, obsPlot.^2/3600, 'blue', 'filled');
% %s2.AlphaData = (obsPlot / max(obsPlot)).^(1/8);
% %s2.MarkerFaceAlpha = 'flat';
% ylim([23.2 240])
% xlim([23.2 240])
% pbaspect([1 1 1])
% grid on
% xlabel('observation (mm/day)')
% ylabel('DL prediction (mm/day)')

%% calculate and plot precision, recall, correlation
precision = zeros(261, 1);
recall = zeros(261, 1);
kcorr = zeros(261, 1);
bds = linspace(10, 140, 261);
for k = 1:261
    truePos = sum(obsMax>=bds(k) & recMax>=bds(k));
    falsPos = sum(obsMax<bds(k) & recMax>=bds(k));
    falsNeg = sum(obsMax>=bds(k) & recMax<bds(k));
    precision(k) = truePos / (truePos + falsPos);
    recall(k) = truePos / (truePos + falsNeg);
    kcorr(k) = corr(obsMax(obsMax>=bds(k)), recMax(obsMax>=bds(k)), 'Type', 'Spearman');
end 

nexttile 
plot(bds, precision, 'k-', bds, recall, 'r-', bds, kcorr, 'b-', 'LineWidth', 2)
xlim([10, 140])
ylim([0 1])
xlabel('threshold (mm)')
ylabel('metrics')
pbaspect([1 1 1])
grid on
legend('precision', 'recall', 'correlation', 'Location', 'southeast')
ttl = title('b)', 'HorizontalAlignment', 'left');
ttl.Units = 'Normalize'; 
ttl.Position(1) = 0; % use negative values (ie, -0.1) to move further left
ttl.HorizontalAlignment = 'left'; 
