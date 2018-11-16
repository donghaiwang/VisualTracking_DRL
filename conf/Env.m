classdef Env
    properties (Constant = true)
        isDebug = true;
        CidiHWaddr = '10:62:e5:04:18:40';
        baseHWaddr = '70:8b:cd:a8:7a:9e';
    end
    
    properties (Dependent)
        workMachine
    end
    
    methods
        function m = get.workMachine(obj)
            [~, curHDaddr] = system("ifconfig -a | grep HWaddr | grep -v 'docker' | awk '{print $5}'");
            curHDaddr = strtrim(curHDaddr);
            if strcmp(curHDaddr, obj.CidiHWaddr) == 1
                m = 'CIDI_WORK';
            elseif strcmp(curHDaddr, obj.baseHWaddr) == 1
                m = 'BASE';
            else
                m = 'OTHER';
            end
        end
    end
end