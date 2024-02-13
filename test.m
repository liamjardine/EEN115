%% Plot graph

topologies = ["./Germany-7nodes/G7-topology.txt", "./Italian-10nodes/IT10-topology.txt"];

% Specify the file path
file_path = topologies(1);

% Read the file into a string and split it into lines
lines = strsplit(fileread(file_path), '\n');

% Remove the first row and join the remaining lines
output_string = strjoin(lines(2:end), '\n');

% Convert the string to a matrix
matrix = str2num(output_string);

% Display the resulting matrix
%disp(matrix);

lx = (length(s));
half = ceil(lx/2);

s = matrix(:,4)';
t = matrix(:,5)';
weights = matrix(:,6)';

G = graph(s(1:half),t(1:half),weights(1:half));

%p = plot(G,'EdgeLabel',G.Edges.Weight);

LWidths = 5*G.Edges.Weight/max(G.Edges.Weight);
p = plot(G,'EdgeLabel',G.Edges.Weight,'LineWidth',LWidths);

[path1,d] = shortestpath(G,1,6);
highlight(p,path1,'EdgeColor','g');

cost = 1;

path_cost = getTransponderCost(path1, cost);

%% Dijkstra

% Specify the file path
file_path = './Germany-7nodes/G7-matrix-1.txt';

% Use the load function to read the matrix
weightedGraph = load(file_path);

source = 1; 

[shortestDistances, predecessors] = dijkstra(weightedGraph, source);

% Display the results
disp('Shortest Distances:');
disp(shortestDistances);
disp('Predecessors:');
disp(predecessors);

%% FSU
