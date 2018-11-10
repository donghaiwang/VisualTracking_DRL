classdef Env
    properties (Constant = true)
        isDebug = false;
        HDaddr = '10:62:e5:04:18:40';
    end
    
    properties (Dependent)
        workMachine
    end
    
    methods
        function m = get.workMachine(obj)
            [~, curHDaddr] = system("ifconfig -a | grep HWaddr | grep -v 'docker' | awk '{print $5}'");
            if strcmp(curHDaddr, obj.HDaddr) == 0
                m = 'CIDI_WORK';
            else
                m = 'OTHER';
            end
        end
    end
end