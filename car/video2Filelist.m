%% convert video to image file and file list
%% to image file in tmp directory
videoFile   = '05_highway_lanechange_25s.mp4';
videoReader = VideoReader(videoFile);
cont = hasFrame(videoReader);
currentStep = 0;
imageListPath = [pwd filesep 'tmp' filesep 'imageList.txt'];
imageListFileID = fopen(imageListPath, 'w');
while cont
    % Update frame counters.
    currentStep = currentStep + 1;
        
    % Read the next frame.
    frame = readFrame(videoReader);
    
    saveImagePath = ['tmp' filesep num2str(currentStep) '.jpg'];
    imwrite(frame, saveImagePath);
    fprintf(imageListFileID, '%s\n', [num2str(currentStep) '.jpg']);
    
    cont = hasFrame(videoReader);
end
fclose(imageListFileID);