% Specify the file path
file_path = './Germany-7nodes/G7-matrix-1.txt';

% Use the load function to read the matrix
matrix = load(file_path);

% Display the matrix
disp('Matrix read from file:');
disp(matrix);