function video_path = get_benchmark_path(bench_name)

if ispc
    dataPathPrefix = 'D:/';     % OVT data directory on Windows
else        % OVT data directory on Linux
    userPath = userpath;
    pathSplited = strsplit(userPath, filesep);
    homeDir = pathSplited(2);
    userName = pathSplited(3);
    dataPathPrefix = [filesep homeDir{1} filesep userName{1} filesep];
end

switch bench_name
    case 'vot15'
        video_path = [dataPathPrefix 'rl/vot2015'];
    case 'vot14'
        video_path = [dataPathPrefix 'rl/vot2014'];
    case 'vot13'
        video_path = [dataPathPrefix 'rl/vot2013'];
end
