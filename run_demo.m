
videos = dir('data');
videoCount = 0;

videoIndexSet = {};
videoNameSet = [];

for i = 1 : length(videos)
    videoItem = videos(i);
    if ~strcmp(videoItem.name, '.') && ~strcmp(videoItem.name, '..') && videoItem.isdir == 1
        videoCount = videoCount+1;
        dispStr = [num2str(videoCount) '  ' videoItem.name];
        disp(dispStr);
        videoIndexSet{videoCount} = videoCount;
        videoNameSet{videoCount} = videoItem.name;
    end
end
videoIndexNameMap = containers.Map(videoIndexSet, videoNameSet);

prompt = 'What is the video index? ';
vidPathID = input(prompt);
vidPath = strcat('data/', videoIndexNameMap(vidPathID));
adnet_demo(vidPath);