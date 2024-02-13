%% Main

% Modulation
modulation_formats = ["SC_DP_QPSK","SC_DP_16QAM","DP_16QAM"];
% Topology
topologies = ["./Germany-7nodes/G7-topology.txt", "./Italian-10nodes/IT10-topology.txt"];
% Traffic
load = 1;
traffics = ["./Germany-7nodes/G7-matrix-" + load + ".txt", "./Italian-10nodes/IT10-matrix-" + load + ".txt"];

% Objects

modulation = Modulation;

% Arguments

modulation = set(modulation, modulation_formats(1));
topology = topologies(1);
traffic = traffics(1);
k = 5;

% Matrix

topology_Matrix = getTopologyMatrix(topology);
traffix_Matrix = getTrafficMatrix(traffic);

% Find K shortest paths

source = 1;
target = 6;

kShortestPaths = kShortestPaths(traffix_Matrix, source, target, k);
