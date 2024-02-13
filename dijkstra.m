function [shortestDistances, predecessors] = dijkstra(graph, source)
    % graph is an adjacency matrix representing the weighted graph
    % source is the starting node for the algorithm
    
    numNodes = size(graph, 1);
    
    % Initialize arrays to store distances and predecessors
    shortestDistances = Inf(1, numNodes);
    predecessors = zeros(1, numNodes);
    
    % Mark the starting node with a distance of 0
    shortestDistances(source) = 0;
    
    % Create a list of unvisited nodes
    unvisitedNodes = 1:numNodes;
    
    while ~isempty(unvisitedNodes)
        % Find the node with the smallest tentative distance
        [~, currentIndex] = min(shortestDistances(unvisitedNodes));
        currentNode = unvisitedNodes(currentIndex);
        
        % Remove the current node from the list of unvisited nodes
        unvisitedNodes(currentIndex) = [];
        
        % Update the distances and predecessors for neighboring nodes
        neighbors = find(graph(currentNode, :) > 0); % Only consider connected nodes
        for neighbor = neighbors
            tentativeDistance = shortestDistances(currentNode) + graph(currentNode, neighbor);
            if tentativeDistance < shortestDistances(neighbor)
                shortestDistances(neighbor) = tentativeDistance;
                predecessors(neighbor) = currentNode;
            end
        end
    end
end