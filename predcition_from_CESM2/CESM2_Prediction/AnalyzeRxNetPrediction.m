precip = ncread("RxNet_Prediction_CESM2_SSP585.nc", "precip");
time = ncread("RxNet_Prediction_CESM2_SSP585.nc", "time");

precipDayMax = squeeze(max(precip, [], [1, 2])); 

iMissing = precipDayMax<0 | precipDayMax>100000 | ismissing(precipDayMax) ;
disp("Totol missing days = " + sum(iMissing))
precipDayMax(iMissing) = NaN;

day = time/365 + 1;

tiledlayout(2, 1, "TileSpacing", "compact")
nexttile
plot(day, precipDayMax)
xlim([2015, 2101])
ylim([0, 350])
xlabel("year")
ylabel("daily maximum (mm)")
pbaspect([2.5, 1, 1])
grid on
text(2016, 330, 'a)', 'FontWeight', 'bold')

% fine top 3 days of each year
iStart = 1;
iEnd = 365;
yearMax = zeros(86, 3);
for k = 1:86
    data = precipDayMax(iStart:iEnd);
    sortData = sort(data, 'descend', 'MissingPlacement', 'last');
    yearMax(k, :) = sortData(1:3);
    iStart = iStart + 365;
    iEnd = iEnd + 365;
end

year = (2015:2100)' + 0.5;
nexttile
plot(year, yearMax, 'o', 'LineWidth', 1, 'MarkerSize', 3);
xlabel("year")
ylabel("annual extremes (mm)")
xlim([2015, 2101])
ylim([0, 350])
pbaspect([2.5, 1, 1])
grid on
text(2016, 330, 'b)', 'FontWeight', 'bold')