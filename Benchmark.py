import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

def rmsa_benchmark_algorithm(topology_graph, traffic_matrix, modulation_formats):
    """
    Basic RMSA benchmark algorithm.

    Args:
        topology_graph (dict): Network topology represented as a graph.
        traffic_matrix (dict): Traffic demands between pairs of nodes.

    Returns:
        dict: Routing, modulation, and spectrum assignment for each demand.
    """
    # Perform basic routing (e.g., shortest path routing)
    #Since we only want the shortest path I guess k=1?
    k = 1
    cost_list = []
    assigned_formats_matrix = []
    for i in range(len(traffic_matrix)):
        for j in range(len(traffic_matrix[i])):
            if traffic_matrix[i][j] != 0:
                source = i + 1  # Assuming 1-based indexing
                target = j + 1  # Assuming 1-based indexing
                traffic_request = traffic_matrix[i][j]  # Get a specific traffic request from the matrix
                shortest_paths = calculate_k_shortest_paths(topology_graph, source, target, k)
                routing_table_matrix = list(nx.shortest_simple_paths(topology_graph, source, target, weight='length'))
                min_cost = float('inf')
                best_path = None
                best_modulation = None
                for path in shortest_paths:
                    print(path)
                    assigned_formats = basic_modulation_assignment(path, topology_graph, modulation_formats,
                                                         traffic_request)
                    # TODO: Is this the part where I need to change? cost<min_cost?
                    for path, _, cost, link in assigned_formats:
                        if cost < min_cost:
                            min_cost = cost
                            best_path = path
                            best_modulation = _
                            best_link = link
                            cost_list.append(min_cost)
                assigned_formats_matrix.append((i, j, best_path, best_modulation, best_link))




                for i, j, path, modulation, link in assigned_formats_matrix:
                    print(f"Traffic request ({i + 1}, {j + 1}):")
                    print("Shortest path:", path)
                    print("Assigned modulation formats:", modulation)
                    # print("used number of link",link)

            total_cost = sum(cost_list)

    spectrum_matrix = create_spectrum_matrix(topology_graph, num_spectrum_slots=320)
    for i, j, path, modulation, link in assigned_formats_matrix:
        least_used_spectrum_assignment(topology_graph, spectrum_matrix, modulation, path, link)

    plt.hist(cost_list, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of transponder costs')
    plt.show()

    return {'routing': shortest_paths, 'modulation': modulation, 'spectrum': spectrum_matrix, 'total cost': total_cost}

def calculate_k_shortest_paths(graph, source, target, k):
    return list(nx.shortest_simple_paths(graph, source, target, weight='length'))[:k]

def basic_modulation_assignment(shortest_path, graph_dict, modulation_formats, traffic_request):
    assigned_modulation_formats = []
    total_length = 0
    for i in range(len(shortest_path) - 1):
        node_a = shortest_path[i]
        node_b = shortest_path[i + 1]
        if node_a in graph_dict and node_b in graph_dict[node_a]:
            length = graph_dict[node_a][node_b]['length']
            total_length += length  # Add the length of the current edge to total length
        else:
            raise Exception(f"Edge ({node_a}, {node_b}) does not exist in the graph.")

    feasible_formats = []
    for format_name, params in modulation_formats.items():
        if total_length <= params['max_length']:
            # print(total_length)
            num_links_required = math.ceil(traffic_request * 10 / params['line_rate'])
            # Calculate the cost for the modulation format considering the number of links required
            total_cost = num_links_required * params['transponder_cost']
            feasible_formats.append((format_name, total_cost, num_links_required))
    if not feasible_formats:
        raise Exception(f"No feasible modulation format for link ({node_a}, {node_b})")

    # Select the format with minimum cost
    selected_format, min_cost, num_links = min(feasible_formats, key=lambda x: x[1])
    assigned_modulation_formats.append((shortest_path, selected_format, min_cost, num_links))
    return assigned_modulation_formats

def find_corresponding_row(topology_graph, link):
    # Find the index of the link in the list of edges
    edges = list(topology_graph.edges())
    try:
        index = edges.index(link)
    except ValueError:
        # Reverse the link and try again (in case the order is different)
        reversed_link = (link[1], link[0])
        index = edges.index(reversed_link) + 11
    return index

def calculate_num_slots_needed(modulation):
    # Calculate the number of slots needed based on modulation
    if modulation == "SC-DP-QPSK":
        return 3
    elif modulation == "DP-QPSK":
        return 3
    elif modulation == "DP-16QAM":
        return 6
    # Add more cases for other modulations as needed
    else:
        return 0  # Default to 0 slots needed if modulation is unknown

def create_spectrum_matrix(topology_graph, num_spectrum_slots):
    num_links = len(topology_graph.edges()) * 2
    return np.zeros((num_links, num_spectrum_slots), dtype=int)

def break_down_path_to_links(path):
    links = []
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        links.append(link)
    return links

def least_used_spectrum_assignment(topology_graph, spectrum_matrix, modulation, path, num_links):
    """
    Assign spectrum slots based on least used spectrum assignment.

    Args:
        topology_graph (NetworkX graph): Network topology graph.
        spectrum_matrix (numpy.ndarray): Spectrum matrix representing spectrum availability.
        modulation (str): Modulation format for the current traffic demand.
        path (list): List of nodes representing the path in the network.

    Returns:
        None. Updates the spectrum_matrix in place.
    """


    # Convert the path to a list of edges
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    print(f"Path edges: {path_edges}")

    links = break_down_path_to_links(path)
    # Find the corresponding row(s) in the spectrum matrix for the links in the path
    row_indices = [find_corresponding_row(topology_graph, link) for link in links]


    # Calculate the number of spectrum units required based on modulation format
    bandwidth_requirement = calculate_num_slots_needed(modulation) * num_links
    print(f"Bandwidth requirement: {bandwidth_requirement}")

    start = find_available_slots(spectrum_matrix, row_indices, bandwidth_requirement)

    # Check if there are enough available spectrum slots to accommodate the bandwidth requirement
    if start is not None:
        print("Enough available slots.")
        # Update the spectrum matrix for the chosen edge to fill up spectrum units
        for i in range(bandwidth_requirement):
            spectrum_matrix[row_indices[0]][start + i] = 1
            #print(f"Assigned slot {i} on edge {links}.")

    else:
        print("Not enough available slots.")

    # Optionally, you can return the chosen edge or any other relevant information
    #return spectrum_matrix


def find_available_slots(spectrum_matrix, row_indices, num_slots_needed):
    #available_slots = []
    num_rows = len(spectrum_matrix)
    num_cols = len(spectrum_matrix[0])

    for col in range(num_cols - num_slots_needed + 1):
        is_available = True
        for row_index in row_indices:

            # Get the corresponding row from the spectrum matrix
            row = spectrum_matrix[row_index]
            if any(row[col + k] != 0 for k in range(num_slots_needed)):
                is_available = False
                break
            if not is_available:
                break

        if is_available:
            return col

    return None  # Return None if no available slots are found


def read_topology_file(file_path):
    # Create an empty graph
    G = nx.Graph()

    # Read the topology file line by line
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            # Split the line into fields
            fields = line.strip().split()

            # Extract relevant information
            node_a = int(fields[3])
            node_b = int(fields[4])
            length = int(fields[5])

            # Add an edge to the graph with the nodes and length
            G.add_edge(node_a, node_b, length=length)

    return G


def plot_topology(graph):
    pos = nx.spring_layout(graph, seed=1)  # Positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=2)

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family="sans-serif")

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'length')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.axis("off")
    plt.show()



def read_traffic_request(file_path):
    traffic_matrix = []

    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            traffic_matrix.append(row)

    return traffic_matrix




#case = 'German'
case = 'Italian'

#Give a number between 1 and 5 to specify a specific traffic file
traffic_case = 5

if case == 'German':
    file_path = 'Germany-7nodes/G7-topology.txt'
    topology_graph = read_topology_file(file_path)
    print('Case: German Topology')
    if traffic_case == 1:
        file_path = 'Germany-7nodes/G7-matrix-1.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 1')
    elif traffic_case == 2:
        file_path = 'Germany-7nodes/G7-matrix-2.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 2')
    elif traffic_case == 3:
        file_path = 'Germany-7nodes/G7-matrix-3.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 3')
    elif traffic_case == 4:
        file_path = 'Germany-7nodes/G7-matrix-4.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 4')
    elif traffic_case == 5:
        file_path = 'Germany-7nodes/G7-matrix-5.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 5')


elif case == 'Italian':
    file_path = 'Italian-10nodes/IT10-topology.txt'
    topology_graph = read_topology_file(file_path)
    print('Case: Italian Topology')
    if traffic_case == 1:
        file_path = 'Italian-10nodes/IT10-matrix-1.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 1')
    elif traffic_case == 2:
        file_path = 'Italian-10nodes/IT10-matrix-2.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 2')
    elif traffic_case == 3:
        file_path = 'Italian-10nodes/IT10-matrix-3.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 3')
    elif traffic_case == 4:
        file_path = 'Italian-10nodes/IT10-matrix-4.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 4')
    elif traffic_case == 5:
        file_path = 'Italian-10nodes/IT10-matrix-5.txt'
        traffic_matrix = read_traffic_request(file_path)
        print('Traffic matrix: 5')


modulation_formats = {
        'SC-DP-QPSK': {'line_rate': 100, 'channel_bandwidth': 37.5, 'max_length': 2000, 'transponder_cost': 1.5},
        'DP-QPSK': {'line_rate': 200, 'channel_bandwidth': 37.5, 'max_length': 700, 'transponder_cost': 2},
        'DP-16QAM': {'line_rate': 400, 'channel_bandwidth': 75, 'max_length': 500, 'transponder_cost': 3.7}
    }


# Run the benchmark algorithm
rmsa_results = rmsa_benchmark_algorithm(topology_graph, traffic_matrix, modulation_formats)

# Print the results
print("Routing:", rmsa_results['routing'])
print("Modulation:", rmsa_results['modulation'])
print("Spectrum:", rmsa_results['spectrum'])



def visualize_spectrum_matrix(spectrum_matrix):
    plt.imshow(spectrum_matrix, cmap='binary', aspect='auto')
    plt.xlabel('Slot Index')
    plt.ylabel('Link Index')
    plt.title('Spectrum Matrix')
    plt.colorbar(label='Occupancy')
    plt.show()
    return

    # Plot the spectrum matrix

# TODO: Don't quite know where we should print out the important parameters,
#  if they belong in the console window or on a graph. And how to pull information from other files to collect them all
total_FSU = np.sum(rmsa_results['spectrum'])
print('Total amount of FSUs: ', total_FSU)

sum_rows_FSU = rmsa_results['spectrum'].sum(axis=1)
highest_FSU = max(sum_rows_FSU)
print('Largest amount of FSUs: ', highest_FSU)

#total_cost = sum(rmsa_results['total_cost'])
print('Total transponder cost: ', rmsa_results['total cost'])


# Assuming 'spectrum_matrix' is your spectrum matrix
visualize_spectrum_matrix(rmsa_results['spectrum'])
plot_topology(topology_graph)
plt.show()