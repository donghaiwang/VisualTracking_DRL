function detector = vehicleDetectorRefineDet(varargin)
%vehicleDetectorRefineDet() Detect vehicles using RefineDet.
%
%  detector = vehicleDetectorFasterRCNN() returns a trained RefineDet
%  object detector for detecting vehicles. RefineDet is a deep learning
%  object detection framework that uses a convolutional neural network
%  (CNN) for detection.
%
%  detector = vehicleDetectorRefineDet(name) returns a vehicle detector
%  based on the specified model name. Valid model names are 'full-view' and
%  'front-rear-view'. The 'full-view' model is trained using unoccluded
%  images of the front, rear, left, and right side of vehicles. The
%  'front-rear-view' model is trained using unoccluded images of the front
%  and rear of vehicles. Both models use a CNN based on a modified version
%  of the highway network.
%
%  A CUDA-capable NVIDIA(TM) GPU with compute capability 3.0 or higher is
%  highly recommended to reduce execution time. Usage of the GPU requires
%  the Parallel Computing Toolbox.
%
%  Example 1: Detect vehicles on a highway
%  ---------------------------------------
%  % Load the pre-trained detector.
%  fasterRCNN = vehicleDetectorFasterRCNN('full-view');
%
%  % Apply the detector.
%  I = imread('highway.png');
%  [bboxes, scores] = detect(fasterRCNN, I);
%  
%  % Annotate detections.   
%  I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
%  figure
%  imshow(I)
%  title('Detected vehicles and detection scores')
%
%  See also fasterRCNNObjectDetector/detect, fasterRCNNObjectDetector,
%           trainFasterRCNNObjectDetector, vehicleDetectorACF, 
%           configureDetectorMonoCamera.

% References:
% -----------
% Ren, Shaoqing, et al. "Faster R-CNN: Towards real-time object detection
% with region proposal networks." Advances in neural information processing
% systems. 2015.

% Copyright 2016 The MathWorks, Inc.

% Add pragma for ML Compiler support.
%#function fasterRCNNObjectDetector 

if nargin > 0
    [varargin{:}] = convertStringsToChars(varargin{:});
end

vision.internal.requiresNeuralToolbox(mfilename);

narginchk(0, 1);

if (isempty(varargin))
    name = 'full-view';
else
    % validate user input
    name = checkModel(varargin{1});
end

detector = loadModel(name);

%--------------------------------------------------------------------------
function [detector, id, name] = loadModel(name)

modelLocation = fullfile(toolboxdir('driving'), 'drivingutilities', 'classifierdata','fasterRCNN');

[name, id] = getModelNameAndID(name);

if id == 1
    modelFile = fullfile(modelLocation, 'front-rear-view.mat');
else
    modelFile = fullfile(modelLocation, 'full-view.mat');
end
data      = load(modelFile);
detector  = data.fasterRCNN;

%--------------------------------------------------------------------------
function [name, id] = getModelNameAndID(name)

switch lower(name)
    case 'front-rear-view'
        name = 'front-rear-view';
        id   = 1;
        
    case 'full-view'
        name = 'full-view';
        id   = 2;
end

%--------------------------------------------------------------------------
function valid = checkModel(model)
valid = validatestring(model,{'front-rear-view','full-view'},...
    mfilename, 'modelName');
