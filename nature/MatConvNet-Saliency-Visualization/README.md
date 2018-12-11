# MatConvNet Saliency Visualization (Merged into 'VisualTracking_DRL'

This is a MatConvNet demo of several saliency visualization methods of ConvNet models.

## References
1. **Error backpropagation:** [Simonyan, K., Vedaldi, A., Zisserman, A.: Deep inside convolutional networks:
Visualising image classification models and saliency maps.](https://arxiv.org/abs/1312.6034)
2. **Class Activation Map:** [Zhou, B., Khosla, A., Lapedriza, A., Oliva, A [[GitHub](https://github.com/jimmie33/Caffe-ExcitationBP]., Torralba, A.: Learning deep features
for discriminative localization.](https://arxiv.org/abs/1512.04150) [[GitHub](https://github.com/metalbubble/CAM)]
3. **Excitation backpropagation:** [Jianming Zhang, Zhe Lin, Jonathan Brandt, Xiaohui Shen, Stan Sclaroff. Top-down Neural Attention by Excitation Backprop.)](http://cs-people.bu.edu/jmzhang/excitationbp.html) [[GitHub](https://github.com/jimmie33/Caffe-ExcitationBP)]

## Prerequisites
1. [MatConvNet](https://github.com/vlfeat/matconvnet).
2. A trained ConvNet model e.g. [ResNet-152](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat) or others [here](http://www.vlfeat.org/matconvnet/models/).

## Installation
1. [Compile MatConvNet](http://www.vlfeat.org/matconvnet/install/).
2. Download the model from the links above, the default model of the demo is the [ResNet-152](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-152-dag.mat) and place in `/Models`.
3. Replace the default **Conv.m** and **Pooling.m** files in the default MatConvNet folder `/MatConvNet/matlab/+dagnn` with the ones included.
