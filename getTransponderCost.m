function path_cost = getTransponderCost(path, cost)
%T_COST Calculates transponder cost for given path
path_cost = (length(path)-1)*cost;
end

