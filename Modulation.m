classdef Modulation
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Line_rate {mustBeNumeric}
        Channel_BW {mustBeNumeric}
        Maximum_length {mustBeNumeric}
        Cost {mustBeNumeric}
    end
    methods
        function obj = set(obj, mod)
            list = zeros(1,4);
            SC_DP_QPSK = [100, 37.5, 2000, 1.5];
            SC_DP_16QAM = [200, 37.5, 700, 2];
            DP_16QAM = [400, 75, 500, 3.7];
        switch mod
            case 'SC_DP_QPSK'
                list = SC_DP_QPSK;
            case 'SC_DP_16QAM'
                list = SC_DP_16QAM;
            case 'DP_16QAM'
                list = DP_16QAM;
            otherwise
                disp('Unknown modulation');
        end
         obj.Line_rate = list(1);
         obj.Channel_BW = list(2);
         obj.Maximum_length = list(3);
         obj.Cost = list(4);
      end
   end
end

