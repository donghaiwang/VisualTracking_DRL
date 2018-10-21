%% display alexnet feature and save it.
clear;

%% save image
fileName = mfilename;
if ~exist(fileName, 'dir')
    mkdir(fileName);
end

%% import net
net = alexnet;

net.Layers;


%% conv1
layer = 2;
name = net.Layers(layer).Name;

size(net.Layers(6, 1).Weights)      % 5x5x48x256

channels = 1:56;

I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1);

conv1Fig = figure;
montage(I);
title(['Layer ',name,' Features'])

saveas(conv1Fig, fullfile(fileName, ['conv' num2str(layer) '.jpg']) );


%% conv2
layer = 6;
channels = 1:30;

I = deepDreamImage(net,layer,channels,...
    'PyramidLevels',1);

conv2Fig = figure;
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features'])

saveas(conv2Fig, fullfile(fileName, ['conv' num2str(layer) '.jpg']) );


%% conv3-5
layers = [10 12 14];
channels = 1:30;

for layer = layers
    I = deepDreamImage(net,layer,channels,...
        'Verbose',false, ...
        'PyramidLevels',1);

    conv3_5 = figure;
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features']);
    
    saveas(conv3_5, fullfile(fileName, ['conv' num2str(layer) '.jpg']) );
end


%% visualize fc
layers = [17 20];

channels = 1:6;

for layer = layers
    I = deepDreamImage(net,layer,channels, ...
        'Verbose',false, ...
        'NumIterations',50);

    fcFig = figure;
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
    
    saveas(fcFig, fullfile(fileName, ['conv' num2str(layer) '.jpg']) );
end


%% finnaly fc
layer = 23;
channels = [9 188 231 563 855 975];

net.Layers(end).ClassNames(channels);

I = deepDreamImage(net,layer,channels, ...
    'Verbose',false, ...
    'NumIterations',50);

finallyFcFig = figure;
montage(I)
name = net.Layers(layer).Name;
title(['Layer ',name,' Features']);

saveas(finallyFcFig, fullfile(fileName, ['conv' num2str(layer) '.jpg']) );





