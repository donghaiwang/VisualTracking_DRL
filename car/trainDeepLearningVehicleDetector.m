%% Train A Deep Learning Vehicle Detector
%% load data set
data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

dataDir = fullfile(toolboxdir('vision'),'visiondata');
vehicleDataset.imageFilename = fullfile(dataDir, vehicleDataset.imageFilename);

idx = floor(0.6 * height(vehicleDataset));
trainingData = vehicleDataset(1:idx, :);
testData = vehicleDataset(idx:end, :);

%% create a convolutional neural network (CNN)
inputLayer = imageInputLayer([32 32 3]);

filterSize = [3 3];
numFilters = 32;

middleLayers = [
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer();
    maxPooling2dLayer(3, 'Stride', 2)
    ];

finalLayers = [
    fullyConnectedLayer(64)
    reluLayer
    % the last fully connected layer.
    fullyConnectedLayer(width(vehicleDataset))
    softmaxLayer
    classificationLayer
    ];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

%% confugre training options
% the mini-batch size must be 1 for Faster R-CNN training.???
% MiniBatchSize 4->128 precision higher
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%% train Faster R-CNN
% 'NumRegionsToSample', [256 128 256 128], ...
doTrainingAndEval = true;
if doTrainingAndEval
    rng(0);
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);
else
    detector = data.detector;
end

I = imread(testData.imageFilename{1});
[bboxes, scores] = detect(detector, I);
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure;
imshow(I);

%% evaluate detector using test set
if doTrainingAndEval
    % Run detector on each image in the test set and collect results.
    resultsStruct = struct([]);
    for i = 1:height(testData)

        % Read the image.
        I = imread(testData.imageFilename{i});

        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);

        % Collect the results.
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end

    % Convert the results into a table.
    results = struct2table(resultsStruct);
else
    results = data.results;
end

expectedResults = testData(:, 2:end);
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

figure;
plot(recall, precision);
xlabel('Recall');
ylabel('Precision');
grid on;
title(sprintf('Average Precision = %.2f', ap));