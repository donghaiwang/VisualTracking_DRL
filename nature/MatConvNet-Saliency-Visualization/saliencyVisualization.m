% saliencyVisualization Saliency demo
% Author: Amir Jamaludin
% Email: amirj@robots.ox.ac.uk
% December 2016; Last revision: 01-Dec-2016
clc
clear
dbstop if error
addpath('Functions')

%% Download model file
if ~exist('Models', 'dir')
    mkdir('Models');
else
    disp('Models directory exist.');
end

modelName = fullfile('Models', 'imagenet-resnet-152-dag.mat');
if ~exist(modelName, 'file')
    modelURL = 'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat';
    websave(modelName, modelURL);
else
    disp('Models exist.');
end

%% MatConvNet path
run(fullfile('../../matconvnet/matlab/vl_setupnn.m'));

%% Initialize Net
modelPath = 'Models/imagenet-resnet-152-dag.mat';
net = dagnn.DagNN.loadobj(load(modelPath));
net.conserveMemory = false;
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
% 2014 Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps 
% method = 'Saliency'; % http://arxiv.org/abs/1312.6034  
method = 'CAM'; % https://arxiv.org/abs/1512.04150
% 2016 Top-down Neural Attention by Excitation Backprop
% method = 'ExBackprop'; % https://arxiv.org/abs/1608.00507  

switch method
    case 'CAM'
        numOfFinalConvFilters = 2048;
    case 'Saliency'
        net = cnn_imagenet_deploy(net);
    case 'ExBackprop'
        net = cnn_imagenet_deploy(net);
        net = changeToConvExBackprop(net);
end
classNumber = 1000;
% net.move('gpu');

%% Prep Image
im = imread('Images/zebra_elephant.jpg');
imSize = size(im);
im_ = single(im);
im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
imResized = im_;
if(strcmp(net.device,'gpu'))
    im_ = gpuArray(im_);
end
im_ = bsxfun(@minus,single(im_),net.meta.normalization.averageImage);

%% Do stuff
% First foward pass & sort the scores
netInput = {'data', im_};
net.eval(netInput);
score = squeeze(gather(net.vars(end).value));
[s,index] = sort(score(:),'descend');

switch method
    case 'CAM'
        finalConvActivation = net.vars(net.getVarIndex('res5cx')).value;
        sizeX = size(finalConvActivation,2);
        sizeY = size(finalConvActivation,1);
        finalConvActivation = reshape(gather(finalConvActivation), [], numOfFinalConvFilters);
        FCWeights = net.params(net.getParamIndex('fc1000_filter')).value;
        FCWeights = reshape(gather(FCWeights),numOfFinalConvFilters,classNumber);
        score = net.vars(net.getVarIndex('prob')).value;
    case 'ExBackprop'
        finalLayer = 'fc1000';
        finalLayerFilter = 'fc1000_filter';
        penultimateLayer = 'pool5';
        contrastOutputLayer = 'res4b23';
end

figure;
subplot(2,3,1);
imshow(im);
for ii = 1:5
d{ii} = net.meta.classes.description{index(ii)};

switch method
    case 'CAM'
        % Change index(ii) to a specific class for that class' saliency
        fCA = finalConvActivation * FCWeights(:, index(ii));
        fCA = reshape(fCA, sizeY, sizeX);
        fCA = imresize(fCA, imSize(1:2));
        mapIm = mat2im(fCA, jet(100), [0 max(fCA(:))]);
        saliencyMap = mapIm*0.5 + (single(im)/255)*0.5;
    case 'Saliency'
        bpropLabel = zeros(1, 1, classNumber, 'single');
        if(strcmp(net.device,'gpu'))
            bpropLabel = gpuArray(bpropLabel);
        end
        % Change index(ii) to a specific class for that class' saliency
        bpropLabel(index(ii)) = 1;
        
        % Net Output
        finalLayer = 'fc1000';
        derOutput = {finalLayer, bpropLabel};
        
        % Forward -> Backward
        net.eval(netInput, derOutput);
        saliencyMap = max(abs(net.vars(1).der), [], 3);
    case 'ExBackprop'
        bpropLabel = zeros(1, 1, classNumber, 'single');
        if(strcmp(net.device,'gpu'))
            bpropLabel = gpuArray(bpropLabel);
        end
        % Change index(ii) to a specific class for that class' saliency
        bpropLabel(index(ii)) = 1;
        
        % Net Output
        derOutput = {finalLayer, bpropLabel};
        
        % Invert Top/Final Layer Weights (non-elephant)
        net.params(net.getParamIndex(finalLayerFilter)).value = net.params(net.getParamIndex(finalLayerFilter)).value .* -1;
        net.eval(netInput,derOutput);   % compute Marginal Winning Probability
        penultimateDerPrime = net.vars(net.getVarIndex(penultimateLayer)).der;
        net.reset
        
        % Invert Back (Go back to before: elephant)
        net.params(net.getParamIndex(finalLayerFilter)).value = net.params(net.getParamIndex(finalLayerFilter)).value .* -1;
        net.eval(netInput,derOutput);   % contrastive MWP
        penultimateDer = net.vars(net.getVarIndex(penultimateLayer)).der;
        net.reset
        
        % Compute Contrastive Signal: Backward From Penultimate Derivative
        contrastDer = penultimateDer - penultimateDerPrime;
        derOutput = {penultimateLayer, contrastDer};    % using the contrastive distinguished encoding rules as backwork input
        
        % Forward -> Backward
        net.eval(netInput,derOutput);
        saliencyMap = max(0,sum(net.vars(net.getVarIndex(contrastOutputLayer)).der,3));        
            
        mapIm = mat2im(saliencyMap, jet(100), [0 max(saliencyMap(:))]);
        mapIm = imresize(mapIm, imSize(1:2));
        saliencyMap = mapIm*0.5 + (single(im)/255)*0.5;
end
subplot(2, 3, ii+1);
imshow(mat2gray(imresize(saliencyMap, imSize(1:2))));
title(sprintf('%s:%.3f', d{ii}, s(ii)));
end