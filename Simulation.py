from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import numpy as np


class TradeAgent(Agent):
    """ This class defines the agents in the model."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        # uncomment for equal initial attributes for all agents (Equal attributes, fully random, targeted attacks simulations)
        self.wealth = 100  # initial wealth
        self.reputation = 15  # initial reputation
        self.trustworthiness = 5  # initial trustworthiness

        # uncomment for random initial attributes for all agents (High and Low attributes simulation)
        # # random choice between two initializations for High and Low attributes simulation
        # if np.random.random() < 0.5:
        #     self.wealth = 50
        #     self.reputation = 10
        #     self.trustworthiness = 6
        # else:
        #     self.wealth = 150
        #     self.reputation = 20
        #     self.trustworthiness = 4


        # Q-learning attributes
        self.q_table = np.zeros((30*2+1,self.model.num_agents))
        self.q_table[:, self.unique_id] = -1
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.epsilon = 0.3 # Exploration rate
        # uncomment for fully random simulation
        # self.epsilon = 1 # Exploration rate


    @property
    def reputation(self):
        """The agent's reputation"""
        return self._reputation

    @reputation.setter
    def reputation(self, value):
        self._reputation = min(max(0, value), 30)  # Ensure reputation is between 0 and 30

    @property
    def trustworthiness(self):
        """The agent's trustworthiness"""
        return self._trustworthiness

    @trustworthiness.setter
    def trustworthiness(self, value):
        self._trustworthiness = min(max(0, value), 10)  # Ensure trustworthiness is between 0 and 10




    def step(self):
        """Agent's step will be to trade with another agent."""

        # Choose an agent to trade with based on Q-values or exploration
        if np.random.random() <= self.epsilon:
            other_agent_id = np.random.choice(np.delete(np.arange(self.model.num_agents), self.unique_id))
            other_agent = self.model.schedule.agents[other_agent_id]
            
        else:
            # Choose an agent to trade with based on Q-values but exclude self from the list of possible agents without using np.delete
            q_example = np.copy(self.q_table)
            state = np.array(self.reputation - self.model.ags_reputations, dtype=int) + 30
            q_val_aux = q_example[state, list(range(self.model.num_agents))]

            other_agent_id = np.random.choice(np.argwhere(q_val_aux == np.max(q_val_aux)).flatten())
            other_agent = self.model.schedule.agents[other_agent_id]

        if self.wealth > 0 and other_agent.wealth > 0:
            self.trade(other_agent)
        else:
            self.q_table_update(self.wealth, other_agent.wealth, other_agent)




    def trade(self, other_agent):
        """ Perform a trade with another agent."""
        # Store initial wealth to compute wealth gained from trade
        initial_wealth = self.wealth
        oa_initial_wealth = other_agent.wealth

        ### uncomment for "uniform" distribution of outcomes
        outcome_probabilities = {
            "mutual_benefit": 0.2 + 0.01*(self.trustworthiness + other_agent.trustworthiness),
            "one_sided_benefit": 0.3 + 0.01* abs(self.reputation - other_agent.reputation),
            "neutral_trade": 0.25,
            "loss_for_both": 0.25
        }

        ### uncomment for "mutual_benefit" is more likely
        # outcome_probabilities = {
        #     "mutual_benefit": 0.7 + 0.01*(self.trustworthiness + other_agent.trustworthiness),
        #     "one_sided_benefit": 0.1 + 0.01* abs(self.reputation - other_agent.reputation),
        #     "neutral_trade": 0.1,
        #     "loss_for_both": 0.1
        # }

        ### uncomment for "one_sided_benefit" is more likely
        # outcome_probabilities = {
        #     "mutual_benefit": 0.1 + 0.01*(self.trustworthiness + other_agent.trustworthiness),
        #     "one_sided_benefit": 0.7 + 0.01* abs(self.reputation - other_agent.reputation),
        #     "neutral_trade": 0.1,
        #     "loss_for_both": 0.1
        # }

        ### uncomment for "neutral trade" is more likely
        # outcome_probabilities = {
        #     "mutual_benefit": 0.1 + 0.01*(self.trustworthiness + other_agent.trustworthiness),
        #     "one_sided_benefit": 0.1 + 0.01* abs(self.reputation - other_agent.reputation),
        #     "neutral_trade": 0.7,
        #     "loss_for_both": 0.1
        # }

        ### uncomment for "loss_for_both" is more likely
        # outcome_probabilities = {
        #     "mutual_benefit": 0.1 + 0.01*(self.trustworthiness + other_agent.trustworthiness),
        #     "one_sided_benefit": 0.1 + 0.01* abs(self.reputation - other_agent.reputation),
        #     "neutral_trade": 0.1,
        #     "loss_for_both": 0.7
        # }

        # Normalize the probabilities so they sum up to 1
        total_prob = sum(outcome_probabilities.values())
        for key in outcome_probabilities:
            outcome_probabilities[key] /= total_prob

        outcome = np.random.choice(list(outcome_probabilities.keys()), p=list(outcome_probabilities.values()))


        if outcome == "mutual_benefit":
            self.wealth += 1
            other_agent.wealth += 1
        elif outcome == "one_sided_benefit":
            if self.reputation > other_agent.reputation:
                if other_agent.wealth > 1:
                    self.wealth += 2
                    other_agent.wealth -= 2
                else:
                    self.wealth += 1
                    other_agent.wealth -= 1
            else:
                if self.wealth > 1:
                    self.wealth -= 2
                    other_agent.wealth += 2
                else:
                    self.wealth -= 1
                    other_agent.wealth += 1
        elif outcome == "neutral_trade":
            pass  # No change in wealth for either agent
        elif outcome == "loss_for_both":
            self.wealth -= 1
            other_agent.wealth -= 1

        self.record_trade(other_agent)

        self.q_table_update(initial_wealth, oa_initial_wealth, other_agent)

        self.model.ags_reputations[self.unique_id] = self.reputation
        self.model.ags_reputations[other_agent.unique_id] = other_agent.reputation
            
        

    def record_trade(self, other_agent):
        """Record the trade in the adjacency matrix."""
        self.model.adjacency_matrix[self.unique_id, other_agent.unique_id] += 1
        self.model.adjacency_matrix[other_agent.unique_id, self.unique_id] += 1

    def q_table_update(self, initial_wealth, oa_initial_wealth, other_agent):
        """Update the Q-table based on the outcome of the trade."""
        reputation_diff = self.reputation - other_agent.reputation

        if self.wealth != 0 and other_agent.wealth != 0:
            # Give feedback based on trade outcome
            if self.wealth > initial_wealth:
                self.give_feedback(other_agent, positive=True)
            elif self.wealth < initial_wealth:
                self.give_feedback(other_agent, positive=False)
            if other_agent.wealth > oa_initial_wealth:
                other_agent.give_feedback(self, positive=True)
            elif other_agent.wealth < oa_initial_wealth:
                other_agent.give_feedback(self, positive=False)

        future_reputation_diff = self.reputation - other_agent.reputation

        # Update Q-value based on trade outcome
        wealth_gained = self.wealth - initial_wealth
        oa_wealth_gained = other_agent.wealth - oa_initial_wealth
        reward = (0.5 * wealth_gained) + (0.3 * other_agent.trustworthiness) + (0.2 * (reputation_diff))
        oa_reward = (0.5 * oa_wealth_gained) + (0.3 * self.trustworthiness) + (0.2 * (-reputation_diff))
        
        old_value = self.q_table[reputation_diff + 30, other_agent.unique_id]
        oa_old_value = other_agent.q_table[-reputation_diff + 30, self.unique_id]
        future_value = np.max(self.q_table[future_reputation_diff + 30,:])
        oa_future_value = np.max(other_agent.q_table[-future_reputation_diff + 30, :])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * future_value)
        oa_new_value = (1 - other_agent.learning_rate) * oa_old_value + other_agent.learning_rate * (oa_reward + other_agent.discount_factor * oa_future_value)

        if self.wealth != 0 and other_agent.wealth != 0:
            self.q_table[reputation_diff + 30,other_agent.unique_id] = new_value
            other_agent.q_table[-reputation_diff + 30, self.unique_id] = oa_new_value
        elif self.wealth == 0 and other_agent.wealth != 0:
            # other_agent.q_table[-reputation_diff + 30, self.unique_id] -= 5
            other_agent.q_table[:, self.unique_id] -= 5
            other_agent.give_feedback(self, positive=False)
        elif self.wealth != 0 and other_agent.wealth == 0:
            # self.q_table[reputation_diff + 30,other_agent.unique_id] -= 5
            self.q_table[:, other_agent.unique_id] -= 5
            self.give_feedback(other_agent, positive=False)
        elif self.wealth == 0 and other_agent.wealth == 0:
            # self.q_table[reputation_diff + 30,other_agent.unique_id] -= 5
            self.q_table[:, other_agent.unique_id] -= 5
            other_agent.q_table[-reputation_diff + 30, self.unique_id] -= 5
            self.give_feedback(other_agent, positive=False)
            other_agent.give_feedback(self, positive=False)

    def give_feedback(self, other_agent, positive=True):
        """Give feedback to another agent based on the outcome of the trade."""
        # Adjust reputation based on feedback
        if positive:
            other_agent.reputation += 1
        else:
            other_agent.reputation -= 1


class TradeModel(Model):
    """ This class defines the economic model."""
    def __init__(self, N, Pr_rand_event=0, attack = False):  
        self.num_agents = N # Number of agents
        self.pr_rand_event = Pr_rand_event # Probability of a random event occuring at each step
        self.attack = attack # Whether or not to attack the network
        self.adjacency_matrix = np.zeros((self.num_agents, self.num_agents))
        self.schedule = RandomActivation(self)  # Randomly activate agents
        self.ags_reputations = np.zeros(self.num_agents)
        
        # Vector to record number of steps since last trade with another agent
        self.steps_since_last_trade = np.zeros(self.num_agents)

        self.last_trade_count = np.zeros((self.num_agents, self.num_agents))
        np.fill_diagonal(self.last_trade_count, -5000) 

        # Create agents
        for i in range(self.num_agents):
            agent = TradeAgent(i, self)
            self.schedule.add(agent)
            self.ags_reputations[i] = agent.reputation

        # DataCollector
        self.datacollector = DataCollector(
             model_reporters={"Adjacency Matrix": self.compute_adjacency_matrix},
            agent_reporters={"Wealth": "wealth", "Reputation": "reputation", "Trustworthiness": "trustworthiness"}
        )



    def step(self, step):
        '''Advance the model by one step.'''
        self.initial_adj_matrix = self.adjacency_matrix.copy()
        self.schedule.step()
        self.datacollector.collect(self)
        self.introduce_event()

        


        if step % 1000 == 0 and step != 0 and self.attack:
            self.attack_network()

        # Comment out for fully random simulation
        # Reduce exploration rate over time
        for agent in self.schedule.agents:
            agent.epsilon *= 0.998

    def introduce_event(self):
        """Introduce a random event at each step with a certain probability."""
        events = ["trustworthiness_fluctuation", "reputation_news_or_scandal", "economic_boom_recession", "random_gift"]
        if np.random.random() < self.pr_rand_event:
            
            chosen_event = np.random.choice(events)

            if chosen_event == "trustworthiness_fluctuation":
                agent = self.random.choice(self.schedule.agents)
                agent.trustworthiness += np.random.choice([-1, 1])

            elif chosen_event == "reputation_news_or_scandal":
                agent = self.random.choice(self.schedule.agents)
                agent.reputation += np.random.choice([-2, 2])

            elif chosen_event == "economic_boom_recession":
                boom_or_recession = np.random.choice([-1, 1])
                for agent in self.schedule.agents:
                    if agent.wealth > 5 and boom_or_recession == -1:
                        agent.wealth -= 5
                    elif boom_or_recession == 1:
                        agent.wealth += 5
            elif chosen_event == "random_gift":
                agent = self.random.choice(self.schedule.agents)
                agent.wealth += 5


    def compute_adjacency_matrix(self):
        """ Update the adjacency matrix."""
        initial_num_trades = np.sum(self.initial_adj_matrix, axis=1)
        

        aux_var = np.sum(self.adjacency_matrix, axis=1) == initial_num_trades

        if any(aux_var):
            for agent in self.schedule.agents:
                if aux_var[agent.unique_id] == True and initial_num_trades[agent.unique_id] != 0 and agent.wealth == 0:
                    self.steps_since_last_trade[agent.unique_id] += 1
                    if self.steps_since_last_trade[agent.unique_id] == 10:
                        self.adjacency_matrix[agent.unique_id, :] = 0
                        self.adjacency_matrix[:, agent.unique_id] = 0
                        self.last_trade_count[agent.unique_id, :] = -5000
                        self.last_trade_count[:, agent.unique_id] = -5000
                elif self.steps_since_last_trade[agent.unique_id] != 0:
                    self.steps_since_last_trade[agent.unique_id] = 0

        
        self.last_trade_count += self.adjacency_matrix == self.initial_adj_matrix
        if np.any(self.last_trade_count == 50):
            for i in np.argwhere(self.last_trade_count == 50):
                self.adjacency_matrix[i[0], i[1]] = 0
                self.last_trade_count[i[0], i[1]] = 0


        return self.adjacency_matrix.copy()   

    def attack_network(self):
        """Attack the network by removing the top 3 nodes with the highest degrees."""
        for i in range(3):
            max_degree_node = np.argmax(np.sum(self.adjacency_matrix, axis=1))
            self.adjacency_matrix[max_degree_node, :] = 0
            self.adjacency_matrix[:, max_degree_node] = 0
            self.last_trade_count[max_degree_node, :] = -5000
            self.last_trade_count[:, max_degree_node] = -5000
            # set wealth and reputation of removed nodes to 0
            self.schedule.agents[max_degree_node].wealth = 0
            self.schedule.agents[max_degree_node].reputation = 0
    
def plot_attributes(model, agents_results, ag_property):
    """Plot the evolution of an attribute over time."""
    plt.figure(figsize=(10, 6))
    for i in range(model.num_agents):
        agent_attribute = agents_results.xs(i, level="AgentID")[ag_property]
        agent_attribute.plot(label= "Agent " + str(i))
    plt.title(ag_property + " Over Time")   
    if model.num_agents < 11: 
        plt.legend()
    plt.xlabel("Steps")
    plt.ylabel(ag_property)
    plt.grid(True)
    plt.savefig(ag_property + '_TA.png')

def plot_trades(adj_matrices, n_agents, agent_id):
    """Plot the number of trades over time for a given agent."""
    plt.figure(figsize=(10, 6))
    for j in range(n_agents):
        if j != agent_id:
                trades = adj_matrices[:,agent_id,j]
                plt.plot(trades, label="Agent " + str(j))
    plt.title("Trades Over Time" + " for Agent " + str(agent_id))
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Number of Trades")
    plt.grid(True)


    
if __name__ == "__main__":
    n_agents = 100 # Number of agents
    N = 5000 # Number of steps
    RE_Prob = 0.0 # Probability of a random event occuring at each step
    attack = True # Whether or not to attack the network
    model = TradeModel(n_agents, RE_Prob, attack)
    for i in range(N):
        if i % 1000 == 0:
            print(i)
        model.step(i)



    results = model.datacollector.get_model_vars_dataframe() # get model results

    # save all adjacency matrices into a 3d tensor
    adj_matrices = np.zeros((N, n_agents, n_agents))
    for i in range(N):
        adj_matrices[i, :, :] = results['Adjacency Matrix'].loc[i]
    np.save('adj_matrices_TA.npy', adj_matrices)
    


    agents_results = model.datacollector.get_agent_vars_dataframe() # get agents results

    # save agents_results to csv
    agents_results.to_csv('agents_results_TA.csv')
    
    # for i in range(n_agents):
    #     plot_trades(adj_matrices, n_agents, i)

    plot_attributes(model, agents_results, "Wealth")
    plot_attributes(model, agents_results, "Reputation")
    # plot_attributes(model, agents_results, "Trustworthiness")
    plt.show()

