function net = changeToConvExBackprop(net)
%CNN_IMAGENET_DEPLOY  Deploy a CNN

isDag = isa(net, 'dagnn.DagNN') ;
if isDag
    dagChangeLayersOfType(net, 'dagnn.Conv') ;
    dagChangeLayersOfType(net, 'dagnn.Pooling') ;
end

% -------------------------------------------------------------------------
function net = dagChangeLayersOfType(net, type)
% -------------------------------------------------------------------------
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
        if(strcmp(type,'dagnn.Pooling'))
            if(strcmp(net.layers(l).block.method,'avg'))
                net.layers(l).block.exBackprop = true ;
            end
        else
            net.layers(l).block.exBackprop = true ;
        end
    end
end


