import numpy as np
import random

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS IS JUST A RANDOM EXAMPLE 
# TO GET A FEELING OF Q-LEARNING, 
# DO NOT TREAT THIS AS USABLE CODE
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class QLearningAgent:
    """
    The "Human Learner" agent. It knows nothing about the world
    and learns by trial and error (receiving rewards).
    
    It uses a "Q-table" to store its knowledge.
    The Q-table maps (state, action) -> expected_future_reward
    """
    
    def __init__(self, agent_config, action_space, state_space):
        print("Q-Learning Agent initialized.")
        
        # --- Config Parameters ---
        self.alpha = agent_config['alpha']
        self.gamma = agent_config['gamma']
        self.epsilon = agent_config['epsilon_start']
        self.epsilon_end = agent_config['epsilon_end']
        
        # Calculate the decay rate
        n_episodes = agent_config.get('n_episodes', 20000) # Get n_episodes from config if available
        self.epsilon_decay = (self.epsilon - self.epsilon_end) / agent_config['epsilon_decay']
        
        # --- Q-Table ---
        self.n_actions = len(action_space)
        
        # The state is (energy_levels, famine_levels)
        # So the Q-table needs to be (energy_levels, famine_levels, n_actions)
        q_table_dims = (state_space[0], state_space[1], self.n_actions)
        
        # Initialize the Q-table with all zeros
        self.q_table = np.zeros(q_table_dims)

    def choose_action(self, state):
        """
        This is the "decision-making" function.
        It uses an epsilon-greedy strategy.
        """
        # --- Exploration vs. Exploitation ---
        if random.uniform(0, 1) < self.epsilon:
            # EXPLORE: Choose a random action
            action = random.randint(0, self.n_actions - 1)
        else:
            # EXPLOIT: Choose the best action from the Q-table
            # state[0] is energy, state[1] is famine
            action = np.argmax(self.q_table[state[0], state[1], :])
            
        return action

    def update(self, state, action, reward, next_state, done):
        """
        This is the "learning" function.
        It updates the Q-table based on the reward received.
        This is the "dopamine" / "Reward Prediction Error" signal.
        """
        
        # --- The Q-Learning Formula (from your PDF) ---
        
        # 1. Get the current Q-value (what we *thought* would happen)
        current_q = self.q_table[state[0], state[1], action]
        
        # 2. Calculate the "target" Q-value (what *actually* happened)
        if done:
            target_q = reward # If the episode is over, the future value is just the final reward
        else:
            # Get the best possible Q-value from the *next* state
            max_future_q = np.max(self.q_table[next_state[0], next_state[1], :])
            target_q = reward + self.gamma * max_future_q
            
        # 3. Calculate the "delta" (the "dopamine" signal / error)
        # delta = (What I Got) - (What I Expected)
        delta = target_q - current_q
        
        # 4. Update the Q-table
        # New Q = Old Q + (learning_rate * delta)
        self.q_table[state[0], state[1], action] = current_q + self.alpha * delta
        
        # --- Epsilon Decay ---
        # After each update, we reduce epsilon slightly to explore less
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
