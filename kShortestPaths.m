function kShortestPaths = kShortestPaths(graph, source, target, k)
%KSHORTESTPATHS Finds the K-shortest paths between two nodes

    kShortestPaths = cell(1, k);
    
    for i = 1:k
        % Find the i-th shortest path
        [shortestDistances, predecessors] = dijkstra(graph, source);
        path = reconstructPath(predecessors, target);
        
        % Add the path to the list
        kShortestPaths{i} = path;
        
        % Modify the graph to avoid the found path
        graph = removePath(graph, path);
    end
end

function path = reconstructPath(predecessors, target)
    path = [];
    current = target;
    
    while current ~= 0
        path = [current path];
        current = predecessors(current);
    end
end

function graph = removePath(graph, path)
    for i = 1:(length(path) - 1)
        graph(path(i), path(i+1)) = Inf; % Increase edge cost to avoid this path
    end
end

