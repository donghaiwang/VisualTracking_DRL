# Tracking Use Deep Reinforcement Learning

- ADNet: Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning [[Paper]](https://drive.google.com/open?id=0B34VXh5mZ22cZUs2Umc1cjlBMFU)
- [[Project Page]](https://sites.google.com/view/cvpr2017-adnet) 
- Email: nongfugengxia@163.com [[main page]](https://github.com/donghaiwang)
  
### Install
- Test on (ubuntu 14.04 and ubuntu 16.04) 64bit, MATLAB 2017a and MATLAB 2018a, use Cuda-8.0 and NVIDIA GTX 1080 TI.
- Run `adnet_compile.m` to compile MatConvNet.
- modify train data path: utils/get_benchmark_path.m
- 'models/imagenet-vgg-m-conv1-3.mat' copy from [[MDNet]](https://github.com/HyeonseobNam/MDNet/blob/master/models/imagenet-vgg-m-conv1-3.mat)

### Run tracking
- run `adnet_demo.m`.

### Other
- [[ADNet]] 0.646   0.880
- [[DRLT]](https://arxiv.org/abs/1701.08936)    0.635
- [[DRL-IS]](https://link.springer.com/content/pdf/10.1007/978-3-030-01240-3_42.pdf)    0.671   0.909
- [[MDNet]](https://github.com/HyeonseobNam/MDNet)  0.678   0.909
- [[SANet]](http://www.dabi.temple.edu/~hbling/publication-selected.htm)   0.661   0.913
- [[ECO]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/ecotrack/index.html) 0.691   0.910
- [[CFNet]](https://blog.csdn.net/discoverer100/article/details/79758131?utm_source=blogxgwz1)

### Prospect
- [[Biology of Vision]](http://www.nature.com/articles/s41586-018-0102-6)
- [[Active Object Tracking]](https://arxiv.org/abs/1808.03405)
- [[Visual Navigation]](https://arxiv.org/abs/1609.05143)



