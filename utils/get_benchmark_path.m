function video_path = get_benchmark_path(bench_name)

switch bench_name
    case 'vot15'
        video_path = '/home/laoli/rl/vot2015';
    case 'vot14'
        video_path = '/home/laoli/rl/vot2014';
    case 'vot13'
        video_path = '/home/laoli/rl/vot2013';
end
