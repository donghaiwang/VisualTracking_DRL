%% convert matlab label data(gTruth) to ground truth recognized by ADNet
clear;clc;
% load('truck_100.mat');
load('truck.mat');      % precision: 0.895857, fps: 15.828065
labelDataTable = timetable2table(gTruth.LabelData);
truckBoxTable = labelDataTable(:, 'truck');
truckBox = table2array(truckBoxTable);

gtFileID = fopen('groundtruth_rect.txt', 'a+');
for i = 1 : length(truckBox)
    tmpBox = truckBox{i};
    fprintf(gtFileID, '%d,%d,%d,%d\n', int32(tmpBox(1)), int32(tmpBox(2)), int32(tmpBox(3)), int32(tmpBox(4)));
end
fclose(gtFileID);
