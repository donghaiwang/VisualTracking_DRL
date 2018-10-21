%% visualize activations of a convolutional neural network

%% load pretrained network and data
net = alexnet;

im = imread(fullfile(matlabroot, 'examples', 'nnet', 'face.jpg'));
imshow(im);
imgSize = size(im);
imgSize = imgSize(1:2);

%% view network architecture
net.Layers;


%% show activations of first convolutional layer
act1 = activations(net, im, 'conv1', 'OutputAs', 'channels');

sz = size(act1);
act1 = reshape(act1, [sz(1) sz(2) 1 sz(3)]);

montage(mat2gray(act1), 'Size', [8 12]);


%% Investigate the Activations in Specific Channels
act1ch32 = act1(:,:,:,32);
act1ch32 = mat2gray(act1ch32);
act1ch32 = imresize(act1ch32,imgSize);
imshowpair(im,act1ch32,'montage')


%% Find the Strongest Activation Channel
[maxValue,maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax,imgSize);
imshowpair(im,act1chMax,'montage')


%% Investigate a Deeper Layer
act5 = activations(net,im,'conv5','OutputAs','channels');
sz = size(act5);
act5 = reshape(act5,[sz(1) sz(2) 1 sz(3)]);
montage(imresize(mat2gray(act5),[48 48]))

[maxValue5,maxValueIndex5] = max(max(max(act5)));
act5chMax = act5(:,:,:,maxValueIndex5);
imshow(imresize(mat2gray(act5chMax),imgSize))

montage(imresize(mat2gray(act5(:,:,:,[3 5])),imgSize))

act5relu = activations(net,im,'relu5','OutputAs','channels');
sz = size(act5relu);
act5relu = reshape(act5relu,[sz(1) sz(2) 1 sz(3)]);
montage(imresize(mat2gray(act5relu(:,:,:,[3 5])),imgSize))


%% Test Whether a Channel Recognizes Eyes
imClosed = imread(fullfile(matlabroot,'examples','nnet','face-eye-closed.jpg'));
imshow(imClosed)

act5Closed = activations(net,imClosed,'relu5','OutputAs','channels');
sz = size(act5Closed);
act5Closed = reshape(act5Closed,[sz(1),sz(2),1,sz(3)]);


channelsClosed = repmat(imresize(mat2gray(act5Closed(:,:,:,[3 5])),imgSize),[1 1 3]);
channelsOpen = repmat(imresize(mat2gray(act5relu(:,:,:,[3 5])),imgSize),[1 1 3]);
montage(cat(4,im,channelsOpen*255,imClosed,channelsClosed*255));
title('Input Image, Channel 3, Channel 5');