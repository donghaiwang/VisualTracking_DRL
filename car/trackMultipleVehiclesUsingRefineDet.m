clear;clc;

addpath('../conf');
env = Env();

%% Configure Vehicle Detector and Multi-Object Tracker

[tracker, positionSelector] = setupTracker();
detector = vehicleDetectorRefineDet();



%% Track Vehicles in a Video
% Setup Video Reader and Player
% videoFile   = fullfile(matlabroot,'toolbox/driving/drivingdata/05_highway_lanechange_25s.mp4');
if env.isDebug
    videoFile   = '05_highway_lanechange_25s.mp4';      % build-in video
else
    if ispc
        videoFile   = 'D:\workspace\data\test\20180505102607869.wmv';   % work on windows
    else
        if strcmp(env.workMachine, 'BASE')
            videoFile = 'tmp/cidi.wmv';
        else
            videoFile = '/data/cidi.wmv';   % linux on cidi work machine
        end
    end
end

videoReader = VideoReader(videoFile);
videoPlayer = vision.DeployableVideoPlayer(); 

videoWriter = VideoWriter(fullfile(tempdir, [datestr(now) '_tracking.avi']));
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
%     confirmedTracks = removeNoisyTracks(confirmedTracks, positionSelector, imageSize);    
    
    % Insert tracking annotations.
    frameWithAnnotations = insertTrackBoxes(frame, confirmedTracks, positionSelector);

    % Display the annotated frame.    
    videoPlayer(frameWithAnnotations);  
    
    writeVideo(videoWriter, frameWithAnnotations);
    
    % Exit the loop if the video player figure is closed by user.
    cont = hasFrame(videoReader) && isOpen(videoPlayer);
end
close(videoWriter);
release(videoPlayer);
detector.close();
clear classes;



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
%     bboxes = detect(detector, frame);
    bufferFilename = 'buffer.jpg';
    imwrite(frame, bufferFilename);
    bboxesList = detector.detect(py.str(bufferFilename));
    for i = 1 : length(bboxesList)
        x1 = double(bboxesList{i}{1});
        y1 = double(bboxesList{i}{2});
        x2 = double(bboxesList{i}{3});
        y2 = double(bboxesList{i}{4});
        bboxes(i, 1) = x1;
        bboxes(i, 2) = y1;
        bboxes(i, 3) = x2 - x1;
        bboxes(i, 4) = y2 - y1;
    end
    delete(bufferFilename);
    
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
    
    % limit the bounding box in the image size
    [imageHeight, imageWidth, ~] = size(I);
    for i = 1 : size(bboxes, 1)
        if bboxes(i, 1) < 0
            bboxes(i, 1) = 0;
        end
        if bboxes(i, 2) < 0
            bboxes(i ,2) = 0;
        end
        if bboxes(i, 3) < 0
            bboxes(i, 3) = 0;
        end
        if bboxes(i, 4) < 0
            bboxes(i, 4) = 0;
        end
    end

    for i = 1:numel(tracks)        
        box = bboxes(i, :);
        
        % Convert to vehicle coordinates using monoCamera object.
%         xyVehicle = imageToVehicle(sensor, [box(1)+box(3)/2, box(2)+box(4)]);
        
        labels{i} = '';
%         labels{i} = sprintf('x=%.1f,y=%.1f',0, 0);
    end
    
    % display bounding box color according tracks.TrackID
    % cell array
%     colorZoo = ["yellow", "blue", "green", "cyan", "red", "magenta"];
    colorZoo = [[220,220,220];...       % Gainsboro
                [139, 0, 0]; ...        % DarkRed
                [255, 218, 185]; ...     % PeachPuff
                [139, 0, 139];      % DarkMagenta
                [240, 255, 240]; ...     % Honeydew
                [0, 139, 139]; ...      % DarkCyan
                [25, 25, 112]; ...      % MidnightBlue
                [0, 0, 139]; ...        % DarkBlue
                [30, 144, 255]; ...     % DodgerBlue
                [139, 123, 139]; ...    % Thistle4
                [96, 123, 139]; ...     % LightSkyBlue4
                [145, 44, 238]; ...     % Purple2
                [0, 255, 255]; ...      % Cyan
                [46, 139, 87]; ...      % SeaGreen4
                [0, 255, 0]; ...        % Green1
                [255, 236, 139]; ...    % LightGoldenrod1
                [255, 255, 0]; ...      % Yellow
                [139, 105, 105]; ...    % RosyBrown4
                [255, 48, 48]; ...      % Firebrick1
                [255, 20, 147]; ...     % DeepPink1
                [144, 238, 144]; ...
            ];    % LightGreen
                       
    displayColor = zeros(length(tracks), 3);
    for i = 1 : length(tracks)
        hashedID = mod(tracks(i).TrackID, length(colorZoo));
        if hashedID == 0
            displayColor(i,:) = colorZoo(int32(length(tracks)/2));
        else
            displayColor(i,:) = colorZoo(hashedID, :);
        end
    end
        I = insertObjectAnnotation(I, 'rectangle', bboxes, labels, 'Color', displayColor, ...
        'FontSize', 10, 'TextBoxOpacity', .8, 'LineWidth', 2);
%     I = insertObjectAnnotation(I, 'rectangle', bboxes, labels, 'Color', displayColor, ...
%         'FontSize', 10, 'TextBoxOpacity', .8, 'LineWidth', 2);
end