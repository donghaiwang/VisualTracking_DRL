videoFReader   = vision.VideoFileReader('atrium.mp4');
depVideoPlayer = vision.DeployableVideoPlayer;

cont = ~isDone(videoFReader);
while cont
    frame = step(videoFReader);
    step(depVideoPlayer, frame);
    cont = ~isDone(videoFReader) && isOpen(depVideoPlayer);
end

release(videoFReader);
release(depVideoPlayer);
