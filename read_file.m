clear
clc
% close all

%% Project settings, load & reformat data

% Parameters we can choose
n = 'Italian';   % 'German' or 'Italian' topology
n_request_integer = 5; % Integer 1 to 5. 1 chooses traffic matrix 1 and so on.
K = 3; % As in K-shortest paths

% Load and reformat data
% load topology and traffic matrices and extract topology data
switch n
    case 'German'
        %load Germany-7nodes/G7-topology.txt
        filepath="Germany-7nodes\G7-topology.txt";
        topology_data=readmatrix(filepath);
        disp('Case: German Topology')

        path = 'Germany-7nodes/';
        for i=1:5           % load all traffic data files
            filename = [path 'G7-matrix-' num2str(i) '.txt'];
            %load (filename)
        end
        switch n_request_integer
            case 1
                Traffic_matrix = readmatrix(filename);
            case 2
                Traffic_matrix = readmatrix(filename);
            case 3
                Traffic_matrix = readmatrix(filename);
            case 4
                Traffic_matrix = readmatrix(filename);
            case 5
                Traffic_matrix = readmatrix(filename);
        end

    case 'Italian'
    %load Italian-10nodes/IT10-topology.txt
    filepath='Italian-10nodes\IT10-topology.txt';

    topology_data = readmatrix(filepath);  
    disp('Case: Italian Topology')

    path = 'Italian-10nodes/';
    for i=1:5       % load all traffic data files
        filename = [path 'IT10-matrix-' num2str(i) '.txt'];
        %load (filename)
    end
    
    switch n_request_integer
        case 1
            Traffic_matrix = readmatrix(filename);
        case 2
            Traffic_matrix = readmatrix(filename);
        case 3
            Traffic_matrix = readmatrix(filename);
        case 4
            Traffic_matrix = readmatrix(filename);
        case 5
            Traffic_matrix = readmatrix(filename);
     end
end

%reformat topology matrix 
topology=reformat_topo(topology_data);
num_nodes = size(traffic_matrix, 1);

% Initialize an empty list to store shortest paths
shortest_paths = cell(num_nodes);

% Step 2 and 3: Calculate shortest paths using Dijkstra's algorithm
for source = 1:num_nodes
    for target = 1:num_nodes
        % Skip if source and target are the same node
        if source == target
            continue;
        end
        
        % Apply Dijkstra's algorithm to find the shortest path
        [shortest_distances, predecessors] = dijkstra(traffic_matrix, source);
        
        % Reconstruct the shortest path from the predecessors
        shortest_paths{source, target} = reconstructPath(predecessors, target);
    end
end



