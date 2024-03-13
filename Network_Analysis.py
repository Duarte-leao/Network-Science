import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from networkx.algorithms import community
from collections import defaultdict

def load_data(attributes_file_name, adjacency_matrices_file_name):
    """ 
    Load the data from the csv file and the numpy file.
    """
    attributes_df = pd.read_csv(attributes_file_name)  
    adjacency_matrices = np.load(adjacency_matrices_file_name)  
    return attributes_df, adjacency_matrices

def plot_numb_nodes(data):
    """
    Plot the number of nodes over time.
    """
    num_nodes_list = []
    for matrix in data:
        G = nx.from_numpy_array(matrix)
        G.remove_nodes_from(list(nx.isolates(G)))
        num_nodes_list.append(G.number_of_nodes())

    
    # Plot the number of nodes and edges over time
    plt.figure(figsize=(10,6))
    plt.plot(num_nodes_list)
    plt.xlabel('Step')
    plt.ylabel('Number of Nodes')
    plt.title('Number of nodes over time')
    plt.grid(True)
    plt.savefig('Num_nodes_FR2.png')
    # plt.show()

def plot_degree_distributions(data):
    """
    Creates an kernel density plot for the degree distributions of multiple time steps of the simulation.
    """

    # Initialize the figure
    plt.figure(figsize=(10,6))

    # For each frame, compute the degree distribution and plot the KDE
    num_frames = list(range(-1,len(data),500))
    num_frames[0]=149
    for frame in num_frames:
        G = nx.from_numpy_array(data[frame])
        G.remove_nodes_from(list(nx.isolates(G)))
        degrees = [d for n, d in G.degree(weight='weight')]
        sns.kdeplot(degrees, label=f"Step {frame+1}", color=plt.cm.turbo(frame/len(data))) # esimation of the probability density function
    plt.title("Evolution of Degree Distributions Over Time")
    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('Degree_distr_FR2.png')
    # plt.show()

def create_animation(data, metric_function, title_format, file_name, *args):
    """
    Create an animation for a given metric function.
    """
    fig, ax = plt.subplots(figsize=(10,6))
    colorbar = None


    def update(frame):
        nonlocal colorbar
        ax.clear()

        if frame >= len(data):
            return
        
        G = nx.from_numpy_array(data[frame])
        G = nx.convert_node_labels_to_integers(G)
        G.remove_nodes_from(list(nx.isolates(G))) # remove isolated nodes


        # Use the passed metric function to get values and labels for plotting
        values, label = metric_function(G, *args)


        pos = nx.spring_layout(G) 

        # Colorbar adjustments
        if colorbar is None:
            sm = ScalarMappable(cmap='turbo', norm=Normalize(vmin=min(values), vmax=max(values)))
            sm.set_array([])
            colorbar = plt.colorbar(sm, ax=ax)
            colorbar.set_label(label)
            nx.draw_networkx_nodes(G, pos, node_color=values, cmap='turbo', node_size=150, vmin=min(values), vmax=max(values))
        else:
            if label == "Degree Centrality":
                vmin = min(min(values), 0.9)
            else:
                vmin = min(values)
            colorbar.update_normal(ScalarMappable(norm=Normalize(vmin=vmin, vmax=max(values)), cmap='turbo'))
            nx.draw_networkx_nodes(G, pos, node_color=values, cmap='turbo', node_size=150, vmin=vmin, vmax=max(values))
        
        
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        ax.set_title(title_format.format(frame+1))

    num_frames = list(range(-1,len(data),500))
    num_frames[0]=0
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000, repeat=False)
    ani.save(file_name, writer='imagemagick', fps=1)
    # plt.show()

# Metric functions
def metric_weighted_degree(G):
    """
    Compute the weighted degree for each node in the graph.
    """
    weighted_degrees = dict(G.degree(weight='weight'))
    weighted_degrees_values = list(weighted_degrees.values())
    return weighted_degrees_values, "Weighted Degree"

def metric_degree_centrality(G):
    """
    Compute the degree centrality for each node in the graph.
    """
    degree_centrality = nx.degree_centrality(G)
    return list(degree_centrality.values()), "Degree Centrality"

def metric_eigenvector_centrality(G):
    """
    Compute the eigenvector centrality for each node in the graph.
    """
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=5000)
    return list(eigenvector_centrality.values()), "Eigenvector Centrality"


def clustering_coefficient(G):
    """
    Compute the weighted clustering coefficient for each node in the graph.
    """
    return np.array(list(nx.clustering(G, weight='weight').values()), dtype=np.float32)

def plot_clustering_coefficients(coefs_matrix):
    """
    Plot the clustering coefficients over time.
    """
    plt.figure(figsize=(10, 6))
    for node_id in range(coefs_matrix.shape[1]):
        plt.plot(range(coefs_matrix.shape[0]), coefs_matrix[:, node_id], label=f'Node {node_id}')
    plt.xlabel('Step')
    plt.ylabel('Clustering Coefficient')
    plt.title('Clustering Coefficients Over Time')
    plt.grid(True)
    plt.savefig('Clustering_coef_FR2.png')
    # plt.show()

def assortativity(G):
    """
    Compute the assortativity coefficient.
    """

    def compute_weighted_attrib_assort(G, attribute, weight='weight'):
        """
        Compute the weighted assortativity for a given node attribute.
        """
        
        # Compute the weighted means of the attribute
        total_weight = sum([d[weight] for u, v, d in G.edges(data=True)])
        mean_u = sum([G.nodes[u][attribute] * d[weight] for u, v, d in G.edges(data=True)]) / total_weight
        mean_v = sum([G.nodes[v][attribute] * d[weight] for u, v, d in G.edges(data=True)]) / total_weight
        
        # Compute the weighted covariance and variances
        cov = sum([d[weight] * (G.nodes[u][attribute] - mean_u) * (G.nodes[v][attribute] - mean_v) for u, v, d in G.edges(data=True)])
        var_u = sum([d[weight] * (G.nodes[u][attribute] - mean_u) ** 2 for u, v, d in G.edges(data=True)])
        var_v = sum([d[weight] * (G.nodes[v][attribute] - mean_v) ** 2 for u, v, d in G.edges(data=True)])
        
        # Compute the weighted assortativity coefficient
        assortativity = cov / np.sqrt(var_u * var_v)
        
        return assortativity
        

    wealth_assortativity = compute_weighted_attrib_assort(G, 'Wealth')
    reputation_assortativity = compute_weighted_attrib_assort(G, 'Reputation')
    trustworthiness_assortativity = compute_weighted_attrib_assort(G, 'Trustworthiness')


    return wealth_assortativity, reputation_assortativity, trustworthiness_assortativity

def plot_assortativities(assortativity_df):
    """
    Plot the assortativity coefficients over time.
    """
    plt.figure(figsize=(10, 6))
    for attribute in ['Wealth_Assortativity', 'Reputation_Assortativity', 'Trustworthiness_Assortativity']:
        plt.plot(assortativity_df['Step'], assortativity_df[attribute], marker='o', label=attribute)
    plt.xlabel('Step')
    plt.ylabel('Assortativity Coefficient')
    plt.title('Attribute Assortativity Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('Assortativity_FR2.png')
    plt.show()


def community_detection(G):
    """
    Detect communities in the graph using the Louvain method.
    """
    communities = community.louvain_communities(G,resolution=1, seed=42)

    return communities
    
def visualize_communities(G, communities, step):
    """
    Visualize the communities detected in the graph.
    """
    plt.figure(figsize=(10,6))
    # Create a color map for communities
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    # sort the community map by key
    community_map = dict(sorted(community_map.items()))
    
    # Draw the graph
    pos = nx.spring_layout(G)
    cmap = plt.cm.get_cmap('turbo', len(communities))
    nx.draw_networkx_nodes(G, pos, node_size=150, cmap=cmap, node_color=list(community_map.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=8, font_color='k')
    # Add a legend for the community colors outside the plot
    plt.legend(handles=[plt.Line2D([], [], color=cmap(i), label=f'Community {i+1}') for i in range(len(communities))])
    plt.title(f'Communities Detected at Step {step+1}')
    plt.savefig(f'Communities_{step+1}_FR2.png')
    # plt.show()


def main(attributes_file_name, adjacency_matrices_file_name):

    # Load the data
    attributes_df, adjacency_matrices = load_data(attributes_file_name, adjacency_matrices_file_name)

    # Plot the number of nodes
    plot_numb_nodes(adjacency_matrices)

    # Plot the degree distributions
    plot_degree_distributions(adjacency_matrices)

    # Create animations for the weighted degree, degree centrality, and eigenvector centrality metrics
    create_animation(adjacency_matrices, metric_weighted_degree, "Time step {} - Weighted Degree", 'w_deg_evol_FR2.gif')
    create_animation(adjacency_matrices, metric_degree_centrality, "Time step {} - Degree Centrality", 'deg_cent_FR2.gif')
    create_animation(adjacency_matrices, metric_eigenvector_centrality, "Time step {} - Eigenvector Centrality", 'eig_cent_FR2.gif')

    # Dictionary to store assortativity coefficients
    assortativity_dict = defaultdict(list)
    # Dictionary to store clustering coefficients
    clustering_coef_matrix = np.zeros((adjacency_matrices.shape[0], adjacency_matrices.shape[1]))

    num_frames = list(range(-1,len(adjacency_matrices),500))
    num_frames[0]=0

    for step, adj_matrix in enumerate(adjacency_matrices):
        if step % 100 == 0:
            print(f"Step {step}")
        G = nx.Graph(adj_matrix)  # Create a NetworkX graph from the adjacency matrix

        # Add node attributes to the graph
        for node_id in G.nodes():
            attributes = attributes_df.loc[(attributes_df['Step'] == step+1) & (attributes_df['AgentID'] == node_id)]
            
            for col in ['Wealth', 'Reputation']:
                nx.set_node_attributes(G, {node_id: attributes[col].values[0]}, col)

        # Compute clustering coefficients
        clustering_coef_matrix[step,:] = clustering_coefficient(G)

        # Compute assortativity coefficients
        wealth_assortativity, reputation_assortativity, trustworthiness_assortativity = assortativity(G)
        assortativity_dict['Step'].append(step)
        assortativity_dict['Wealth_Assortativity'].append(wealth_assortativity)
        assortativity_dict['Reputation_Assortativity'].append(reputation_assortativity)
        # assortativity_dict['Trustworthiness_Assortativity'].append(trustworthiness_assortativity) # uncomment only for H&L simulation

        if step in num_frames:
            G.remove_nodes_from(list(nx.isolates(G)))
            # Community Detection
            communities = community_detection(G)
            # Visualize the communities detected in the graph
            visualize_communities(G, communities, step)


    # Plot the clustering coefficients over time
    plot_clustering_coefficients(clustering_coef_matrix)
    # save clustering coefficients matrix
    np.save('clustering_coef_matrix_FR2.npy', clustering_coef_matrix)

    # Convert to DataFrame for easier plotting
    assortativity_df = pd.DataFrame(assortativity_dict)
    # save assortativity
    assortativity_df.to_csv('assortativity_FR2.csv')

    # Plot the assortativity coefficients over time
    plot_assortativities(assortativity_df)


if __name__ == "__main__":
    main("agents_results_FR2.csv", "adj_matrices_FR2.npy")


