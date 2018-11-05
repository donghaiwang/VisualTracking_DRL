%% Configure Vehicle Detector and Multi-Object Tracker
% detector = vehicleDetectorACF('full-view');
detector = vehicleDetectorFasterRCNN('full-view');  % more slow, high precision, model path: /home/dong/matlab/toolbox/driving/drivingutilities/classifierdata/fasterRCNN/full-view.mat (front-rear-view.mat)
[tracker, positionSelector] = setupTracker();


%% Track Vehicles in a Video
% Setup Video Reader and Player
% videoFile   = fullfile(matlabroot,'toolbox/driving/drivingdata/05_highway_lanechange_25s.mp4');
videoFile   = '05_highway_lanechange_25s.mp4';
% videoFile   = 'D:\workspace\data\test\20180505102607869.wmv';
videoReader = VideoReader(videoFile);
videoPlayer = vision.DeployableVideoPlayer(); 

videoWriter = VideoWriter('tmp/tracking.avi');
open(videoWriter);

currentStep = 0;
snapshot = [];
snapTimeStamp = 120;
cont = hasFrame(videoReader);
while cont
    % Update frame counters.
    currentStep = currentStep + 1;
        
    % Read the next frame.
    frame = readFrame(videoReader);
    imageSize = size(frame);  imageSize = imageSize(1:2);
    
    % Run the detector and package the returned results into an object
    % required by multiObjectTracker.  You can find the |detectObjects|
    % function at the end of this example.
    detections = detectObjects(detector, frame, currentStep);
       
    % Using the list of objectDetections, return the tracks, updated for
    % 'currentStep' time.
    confirmedTracks = updateTracks(tracker, detections, currentStep);
    
    % Remove the tracks for vehicles that are far away.(d.sensor.Intrinsics.ImageSize: 480x640)
    confirmedTracks = removeNoisyTracks(confirmedTracks, positionSelector, imageSize);    
    
    % Insert tracking annotations.
    frameWithAnnotations = insertTrackBoxes(frame, confirmedTracks, positionSelector);

    % Display the annotated frame.    
    videoPlayer(frameWithAnnotations);  
    
    writeVideo(videoWriter, frameWithAnnotations);
    
    % Exit the loop if the video player figure is closed by user.
    cont = hasFrame(videoReader) && isOpen(videoPlayer);
end
close(videoWriter);



%%
function [tracker, positionSelector] = setupTracker()
    % Create the tracker object.
    tracker = multiObjectTracker('FilterInitializationFcn', @initBboxFilter, ...
        'AssignmentThreshold', 50, ...
        'NumCoastingUpdates', 5, ... 
        'ConfirmationParameters', [3 5]);

    % The State vector is: [x; vx; y; vy; w; vw; h; vh]
    % [x;y;w;h] = positionSelector * State
    positionSelector = [1 0 0 0 0 0 0 0; ...
                        0 0 1 0 0 0 0 0; ...
                        0 0 0 0 1 0 0 0; ...
                        0 0 0 0 0 0 1 0]; 
end

% 
function filter = initBboxFilter(Detection)
% Step 1: Define the motion model and state.
%   Use a constant velocity model for a bounding box on the image.
%   The state is [x; vx; y; vy; w; wv; h; hv]
%   The state transition matrix is: 
%       [1 dt 0  0 0  0 0  0;
%        0  1 0  0 0  0 0  0; 
%        0  0 1 dt 0  0 0  0; 
%        0  0 0  1 0  0 0  0; 
%        0  0 0  0 1 dt 0  0; 
%        0  0 0  0 0  1 0  0;
%        0  0 0  0 0  0 1 dt; 
%        0  0 0  0 0  0 0  1]
%   Assume dt = 1. This example does not consider time-variant transition
%   model for linear Kalman filter.
    dt = 1;
    cvel =[1 dt; 0 1];
    A = blkdiag(cvel, cvel, cvel, cvel);
 
% Step 2: Define the process noise. 
%   The process noise represents the parts of the process that the model
%   does not take into account. For example, in a constant velocity model,
%   the acceleration is neglected.
    G1d = [dt^2/2; dt];
    Q1d = G1d*G1d';
    Q = blkdiag(Q1d, Q1d, Q1d, Q1d);
 
% Step 3: Define the measurement model.
%   Only the position ([x;y;w;h]) is measured.
%   The measurement model is
    H = [1 0 0 0 0 0 0 0; ...
         0 0 1 0 0 0 0 0; ...
         0 0 0 0 1 0 0 0; ...
         0 0 0 0 0 0 1 0];
 
% Step 4: Map the sensor measurements to an initial state vector.
%   Because there is no measurement of the velocity, the v components are
%   initialized to 0:
    state = [Detection.Measurement(1); ...
             0; ...
             Detection.Measurement(2); ...
             0; ...
             Detection.Measurement(3); ...
             0; ...
             Detection.Measurement(4); ...
             0];
 
% Step 5: Map the sensor measurement noise to a state covariance.
%   For the parts of the state that the sensor measured directly, use the
%   corresponding measurement noise components. For the parts that the
%   sensor does not measure, assume a large initial state covariance. That way,
%   future detections can be assigned to the track.
    L = 100; % Large value
    stateCov = diag([Detection.MeasurementNoise(1,1), ...
                     L, ...
                     Detection.MeasurementNoise(2,2), ...
                     L, ...
                     Detection.MeasurementNoise(3,3), ...
                     L, ...
                     Detection.MeasurementNoise(4,4), ...
                     L]);
 
% Step 6: Create the correct filter.
%   In this example, all the models are linear, so use trackingKF as the
%   tracking filter.
    filter = trackingKF(...
        'StateTransitionModel', A, ...
        'MeasurementModel', H, ...
        'State', state, ...
        'StateCovariance', stateCov, ... 
        'MeasurementNoise', Detection.MeasurementNoise, ...
        'ProcessNoise', Q);
end

% 
function detections = detectObjects(detector, frame, frameCount)
    % Run the detector and return a list of bounding boxes: [x, y, w, h]
    bboxes = detect(detector, frame);
    
    % Define the measurement noise.
    L = 100;
    measurementNoise = [L 0  0  0; ...
                        0 L  0  0; ...
                        0 0 L/2 0; ...
                        0 0  0 L/2];
                    
    % Formulate the detections as a list of objectDetection reports.
    numDetections = size(bboxes, 1);
    detections = cell(numDetections, 1);                      
    for i = 1:numDetections
        detections{i} = objectDetection(frameCount, bboxes(i, :), ...
            'MeasurementNoise', measurementNoise);
    end
end

%
function tracks = removeNoisyTracks(tracks, positionSelector, imageSize)

    if isempty(tracks)
        return
    end
    
    % Extract the positions from all the tracks.
    positions = getTrackPositions(tracks, positionSelector);
    % The track is 'invalid' if the predicted position is about to move out
    % of the image, or if the bounding box is too small. Typically, this
    % implies the vehicles is far away.
    invalid = ( positions(:, 1) < 1 | ...
                positions(:, 1) + positions(:, 3) > imageSize(2) | ...
                positions(:, 3) <= 20 | ...
                positions(:, 4) <= 20 );
    tracks(invalid) = [];
end

function I = insertTrackBoxes(I, tracks, positionSelector)

    if isempty(tracks)
        return
    end

    % Allocate memory.
    labels = cell(numel(tracks), 1);
    % Retrieve positions of bounding boxes.
    bboxes = getTrackPositions(tracks, positionSelector);

    for i = 1:numel(tracks)        
        box = bboxes(i, :);
        
        % Convert to vehicle coordinates using monoCamera object.
%         xyVehicle = imageToVehicle(sensor, [box(1)+box(3)/2, box(2)+box(4)]);
        
        labels{i} = '';
%         labels{i} = sprintf('x=%.1f,y=%.1f',0, 0);
    end
    
    % display bounding box color according tracks.TrackID
    % cell array
    colorZoo = ["blue", "green", "cyan", "red", "magenta", "black", "white"];
    displayColor = {};
    for i = 1 : length(tracks)
        displayColor{i} = colorZoo(mod(tracks(i).TrackID, length(colorZoo))+1);
    end
    I = insertObjectAnnotation(I, 'rectangle', bboxes, labels, 'Color', displayColor, ...
        'FontSize', 10, 'TextBoxOpacity', .8, 'LineWidth', 2);
end