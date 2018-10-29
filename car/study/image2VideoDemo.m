workingDir = tempname;
mkdir(workingDir);
mkdir(workingDir, 'images');

shuttleVideo = VideoReader('shuttle.avi');

% Create the image sequence
ii = 1;
while hasFrame(shuttleVideo)
    img = readFrame(shuttleVideo);
    filename = [sprintf('%03d', ii) '.jpg'];
    fullname = fullfile(workingDir, 'images', filename);
    imwrite(img, fullname);
    ii = ii+1;
end

% find images file names
imagesName = dir(fullfile(workingDir, 'images', '*.jpg'));
imagenames = {imagesName.name}';

% create new video with image sequence
outputVideo = VideoWriter(fullfile(workingDir, 'shuttle_out.avi'));
outputVideo.FrameRate = shuttleVideo.FrameRate;
open(outputVideo);

for ii = 1:length(imagenames)
    img = imread(fullfile(workingDir, 'images', imagenames{ii}));
    writeVideo(outputVideo, img);
end

close(outputVideo);

% view the final video
shuttleAvi = VideoReader(fullfile(workingDir, 'shuttle_out.avi'));

ii = 1;
while hasFrame(shuttleAvi)
    mov(ii) = im2frame(readFrame(shuttleAvi));
end

figure;
imshow(mov(1).cdata, 'Border', 'tight');

movie(mov, 1, shuttleAvi.FrameRate);