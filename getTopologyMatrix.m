function matrix = getTopologyMatrix(file_path)
%GETTOPOLOGYMATRIX Turns textfile to matrix

% Read the file into a string and split it into lines
lines = strsplit(fileread(file_path), '\n');

% Remove the first row and join the remaining lines
output_string = strjoin(lines(2:end), '\n');

% Convert the string to a matrix
matrix = str2num(output_string);

end

