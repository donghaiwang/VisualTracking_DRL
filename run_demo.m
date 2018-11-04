
videos = dir('data');
videoCount = 0;

videoIndexSet = {};
videoNameSet = [];

for i = 1 : length(videos)      % record all the video index and video name
    videoItem = videos(i);
    if ~strcmp(videoItem.name, '.') && ~strcmp(videoItem.name, '..') && videoItem.isdir == 1
        videoCount = videoCount+1;
        dispStr = [num2str(videoCount) '  ' videoItem.name];
        disp(dispStr);
        videoIndexSet{videoCount} = videoCount;
        videoNameSet{videoCount} = videoItem.name;
    end
end

if isempty(videoIndexSet) && isempty(videoNameSet)
    disp('There is no video file in data directory');
    return;
else
    videoIndexNameMap = containers.Map(videoIndexSet, videoNameSet);
end


disp('0  tracking all the video in data directory');
prompt = 'What is the video index? ';
vidPathID = input(prompt);      % input the tracking video id
if vidPathID == 0       % tracking all video in the data directory if input 0
    for i = 1 : videoCount
        vidPath = strcat('data/', videoIndexNameMap(i));
        adnet_demo(vidPath);
    end
else
    vidPath = strcat('data/', videoIndexNameMap(vidPathID));
    adnet_demo(vidPath);
end
