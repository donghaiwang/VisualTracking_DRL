%% Automate Attributes of Labeled Objects
%% detect vehicles from a monocular camera
vidObj = VideoReader('05_highway_lanechange_25s.mp4');
vidObj.CurrentTime = 0.1;
I = readFrame(vidObj);

data = load('FCWDemoMonoCameraSensor.mat', 'sensor');
sensor = data.sensor;

detector = vehicleDetectorACF();

vehicleWidth = [1.5, 2.5];

detector = configureDetectorMonoCamera(detector, sensor, vehicleWidth);

[bboxes, ~] = detect(detector, I);
Iout = insertShape(I, 'rectangle', bboxes);
figure;
imshow(Iout);
title('Detected Vehicles');

%% estimate distance to detected vehicles
midPtsImg = [bboxes(:,1)+bboxes(:,3)/2 bboxes(:,2)+bboxes(:,4)/2];
midPtsWorld = imageToVehicle(sensor, midPtsImg);
x = midPtsWorld(:, 1);
y = midPtsWorld(:, 2);
distance = sqrt(x.^2 + y.^2);

% Display vehicle bounding boxes and annotate them with distance in meters.
distanceStr = cellstr([num2str(distance) repmat(' m', [length(distance) 1])]);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, distanceStr);
imshow(Iout);
title('Distance of Vehicles from Camera');

%% use the vehicle detection and distance estimation automation class in the app
mkdir('+vision/+labeler');
% copyfile(fullfile(matlabroot, 'examples', 'driving', 'main', 'VehicleDetectionAndDistanceEstimation.m'), '+vision/+labeler');  % only contains in matlab2018b
load('FCWDemoMonoCameraSensor.mat', 'sensor');
groundTruthLabeler 05_highway_lanechange_25s.mp4