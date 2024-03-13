# Trade Simulation and Network Analysis

This project contains code for both simulating a trade network using agent-based modeling (`Simulation.py`) and analyzing the resulting trade network over time(`Network_Analysis.py`).

## Simulation.py

This code provides a simulation model for agents trading with each other. The model is built using the Mesa framework, a Python library for agent-based modeling.

### Overview

#### TradeAgent Class

This class defines individual agents in the model. Each agent has attributes like wealth, reputation, trustworthiness, and a Q-learning table to learn optimal trading strategies.

- **Attributes**:
  - `wealth`: Amount of wealth the agent has.
  - `reputation`: The reputation score of the agent.
  - `trustworthiness`: How trustworthy the agent is.
  - `q_table`: Q-learning table for the agent to learn optimal trading strategies.
  - `learning_rate`, `discount_factor`, `epsilon`: Parameters for the Q-learning algorithm.

- **Methods**:
  - `step()`: Defines the agent's actions in each step, primarily trading with another agent.
  - `trade()`: Execute a trade with another agent.
  - `record_trade()`: Record the trade in the adjacency matrix.
  - `q_table_update()`: Update the Q-table based on the outcome of the trade.
  - `give_feedback()`: Provide feedback to another agent based on the outcome of the trade.

#### TradeModel Class

This class defines the overall economic model.

- **Attributes**:
  - `num_agents`: Total number of agents in the model.
  - `pr_rand_event`: Probability of a random event occurring at each step.
  - `attack`: Whether or not to attack the network.
  - `adjacency_matrix`: Matrix to record trades between agents.
  - `schedule`: Schedule for agent activation.
  - `ags_reputations`: Vector to store agents' reputations.

- **Methods**:
  - `step()`: Advance the model by one step.
  - `introduce_event()`: Introduce a random event at each step with a certain probability.
  - `compute_adjacency_matrix()`: Update the adjacency matrix.
  - `attack_network()`: Attack the network by removing nodes with the highest degrees.

### Execution

1. Import necessary libraries and the provided code.
2. Initialize the `TradeModel` with the desired number of agents.
3. Run the model for a specified number of steps.

### Key Features

- **Q-learning**: Agents use Q-learning to adapt and optimize their trading strategies over time.
- **Random Events**: The model can introduce random events that affect agents' attributes.
- **Network Attack**: The model can simulate an attack on the network by removing nodes with the highest degrees.

### Dependencies

- numpy version: 1.23.3
- matplotlib version: 3.6.0
- mesa version: 2.1.1

## Network_Analysis.py

### Overview

The provided code is designed to analyze and visualize the evolution of a trade network over time. It uses various metrics and visualization techniques to understand the dynamics of the network, including node attributes, degree distributions, centrality measures, clustering coefficients, and community detection.

### Dependencies

- numpy version: 1.23.3
- matplotlib version: 3.6.0
- pandas version: 1.5.0
- seaborn version: 0.12.0
- networkx version: 2.8.7

### Key Functions

1. **load_data**: Loads the data from a CSV file containing agent attributes and a numpy file containing adjacency matrices representing the network at different time steps.
2. **plot_numb_nodes**: Plots the number of nodes in the network over time.
3. **plot_degree_distributions**: Creates a kernel density plot for the degree distributions of multiple time steps of the simulation.
4. **create_animation**: Generates an animation for a given metric function, visualizing its evolution over time.
5. **metric functions**: Functions like `metric_weighted_degree`, `metric_degree_centrality`, and `metric_eigenvector_centrality` compute various metrics for nodes in the network.
6. **clustering_coefficient**: Computes the weighted clustering coefficient for each node in the graph.
7. **plot_clustering_coefficients**: Plots the clustering coefficients of nodes over time.
8. **assortativity**: Computes the assortativity coefficient based on various node attributes.
9. **plot_assortativities**: Plots the assortativity coefficients over time.
10. **community_detection**: Detects communities in the graph using the Louvain method.
11. **visualize_communities**: Visualizes the detected communities in the network.
12. **main**: The main function that orchestrates the loading of data, computation of metrics, and visualization.

### Execution

To run the code, ensure that you have the necessary dependencies installed. Then, execute the script. The main function requires two file names as arguments:

- `attributes_file_name`: The name of the CSV file containing agent attributes, previously saved in the simulation.
- `adjacency_matrices_file_name`: The name of the numpy file containing adjacency matrices, previously saved in the simulation.


### Outputs

The code will generate various plots and animations, including:

- Number of nodes over time.
- Degree distributions.
- Animations for weighted degree, degree centrality, and eigenvector centrality metrics.
- Clustering coefficients over time.
- Assortativity coefficients over time.
- Visualizations of detected communities at specific time steps.

These outputs will be saved as image and animation files in the current directory.
