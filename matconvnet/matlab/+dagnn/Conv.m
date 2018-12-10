classdef Conv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
    exBackprop = false
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        if ~obj.hasBias, params{2} = [] ; end
        if(~obj.exBackprop)
            [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
                gpuArray(inputs{1}), gpuArray(params{1}), gpuArray(params{2}), gpuArray(derOutputs{1}), ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
        else            
            wPlus = max(0,params{1});
            b = params{2};

            Abot = inputs{1};
            Ptop = derOutputs{1};

            % Forward Pass
            Atop = vl_nnconv(...
                Abot, wPlus, [], ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
            
            Y = gpuArray(zeros(size(Atop),'single'));
            nonZeroXIdx = logical(gather(Atop ~= 0));
            Y(nonZeroXIdx) = Ptop(nonZeroXIdx) ./ Atop(nonZeroXIdx);
            
            % Backward Pass
            [Pbot, derParams{1}, derParams{2}] = vl_nnconv(...
                gpuArray(Abot), gpuArray(wPlus), [], Y, ...
                'pad', obj.pad, ...
                'stride', obj.stride, ...
                'dilate', obj.dilate, ...
                obj.opts{:}) ;
            
            derInputs{1} = Abot .* Pbot;
        end
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = Conv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
