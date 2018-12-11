classdef Pooling < dagnn.Filter
  properties
    method = 'max'
    poolSize = [1 1]
    opts = {'cuDNN'}
    exBackprop = false
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnpool(inputs{1}, self.poolSize, ...
                             'pad', self.pad, ...
                             'stride', self.stride, ...
                             'method', self.method, ...
                             self.opts{:}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
        if(~self.exBackprop)
            derInputs{1} = vl_nnpool(gpuArray(inputs{1}), self.poolSize, gpuArray(derOutputs{1}), ...
                'pad', self.pad, ...
                'stride', self.stride, ...
                'method', self.method, ...
                self.opts{:}) ;
            derParams = {} ;
        else
            Abot = inputs{1};
            Ptop = derOutputs{1};
            
            Atop = vl_nnpool(Abot, self.poolSize, ...
                'pad', self.pad, ...
                'stride', self.stride, ...
                'method', self.method, ...
                self.opts{:}) ;
            
%             Y = zeros(size(Atop),'single');
            Y = gpuArray(zeros(size(Atop),'single'));
            nonZeroXIdx = logical(gather(Atop ~= 0));
            Y(nonZeroXIdx) = Ptop(nonZeroXIdx) ./ Atop(nonZeroXIdx);
            
            Pbot = vl_nnpool(gpuArray(Abot), self.poolSize, Y, ...
                'pad', self.pad, ...
                'stride', self.stride, ...
                'method', self.method, ...
                self.opts{:}) ;

            derInputs{1} = Abot .* Pbot;
            derParams = {} ;
        end
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
