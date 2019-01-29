% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% compile cpp files
% change the include and lib path if necessary
function compile
    
PROJECT_ROOT = fileparts(pwd);
addpath(fullfile(PROJECT_ROOT, 'conf'));    % add project configuration path

if isunix
    include = ' -I/usr/local/include/opencv/ -I/usr/local/include/ -I/usr/include/opencv/';
    lib = ' -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video';
    eval(['mex lk.cpp -O' include lib]);
else
    include = ' -ID:\workspace\lib\opencv-2.4.13\include -ID:\workspace\lib\opencv-2.4.13\include\opencv -ID:\workspace\lib\opencv-2.4.13\include\opencv2';
    libpath = 'D:\workspace\lib\opencv-2.4.13\lib\';
%     libpath = 'D:\workspace\rl\DRLT\VisualTracking_DRL\MDP_Tracking\';
    opencvLibFiles = dir([libpath '*.lib']);
    lib = [];
    for i = 1 : length(opencvLibFiles)
        lib = [lib ' ' libpath opencvLibFiles(i).name];
    end
    if Env.isDebug
        eval(['mex lk.cpp -g' include lib]);  % Use this option to compile with optimization(-O).
    else
        eval(['mex lk.cpp -O' include lib]);
    end
    
end

if Env.isDebug
    mex -g distance.cpp
    mex -g imResampleMex.cpp
    mex -g warp.cpp
else
    mex -O distance.cpp
    mex -O imResampleMex.cpp
    mex -O warp.cpp
end


disp('Compilation finished.');