% Specify the file path
file_path = './Germany-7nodes/G7-topology.txt';

% Read the file into a string and split it into lines
lines = strsplit(fileread(file_path), '\n');

% Remove the first row and join the remaining lines
output_string = strjoin(lines(2:end), '\n');

% Convert the string to a matrix
matrix = str2num(output_string);

% Display the resulting matrix
disp(matrix);

s = matrix(:,4)';
t = matrix(:,5)';
weights = matrix(:,6)';

G = graph(s,t,weights);
plot(G,'EdgeLabel',G.Edges.Weight)