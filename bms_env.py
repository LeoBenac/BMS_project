import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BMSenv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    

    MAX_VOLTAGE = 4.2
    MIN_VOLTAGE = 2.2
    INIT_SOC = 0.9
    I_CURRENT = 0.22
    TIMESTEP = 1e-3



    
    def __init__(self, num_cells: int = 2,  k_tanh_params : list = [0.75, 2.0],
                   Q_cells: list = [2.35, 2.35], w_reward= 100.0):
        
        super(BMSenv, self).__init__()

        # Define action and observation space
        self.observation_space = spaces.Box(low= self.MIN_VOLTAGE, high=self.MAX_VOLTAGE , shape=(num_cells,), dtype=np.float64)
        self.action_space = spaces.Discrete(2**num_cells)

        self.num_cells = num_cells

        assert len(k_tanh_params) == self.num_cells, "Size of k_tanh_params must match num_cells"
        assert len(Q_cells) == self.num_cells, "Size of Q_cells must match num_cells"

        self.k_tanh_params = np.array(k_tanh_params)
        self.Q_cells = np.array(Q_cells)
       
        self._initialize_state()
        self.w_reward = w_reward


    def _initialize_state(self) -> None:
        self.state = np.array([self.MAX_VOLTAGE - 1e-5]* self.num_cells)
        self.state_soc = self.map_voltage_to_soc(self.state, self.k_tanh_params)


    def get_state(self) -> np.array:
        return self.state
    

    def map_voltage_to_soc(self, voltage: float, k: float) -> float:
        """
        Map a voltage value to a state of charge (SOC) using a parametrized tanh function.
        
        Parameters:
        voltage (float): The voltage value to be mapped
        k (float): The steepness parameter for the tanh function
        
        Returns:
        float: The corresponding SOC value
        """
        # Scale and shift tanh to map voltage in [2.2, 4.2] to SOC in [0.1, 0.9]
        return 0.5 * (np.tanh(k * (voltage - 3.2) / 0.5) + 1) * 0.8 + 0.1
    
    def map_soc_to_voltage(self, soc: float, k: float) -> float:
        """
        Map a state of charge (SOC) value to a voltage using the inverse of the parametrized tanh function.
        
        Parameters:
        soc (float): The SOC value to be mapped
        k (float): The steepness parameter for the tanh function
        
        Returns:
        float: The corresponding voltage value
        """
        # Inverse of the SOC mapping function
        # return 3.2 + 0.5 * np.arctanh((soc - 0.1) / 0.8 * 2 - 1) / k
        soc_clipped = np.clip((soc - 0.1) / 0.8 * 2 - 1, -1 + 1e-10, 1 - 1e-10)  # Clip values to avoid invalid arctanh
        return 3.2 + 0.5 * np.arctanh(soc_clipped) / k
    

    def int_action_to_switch_action(self, action: int) -> np.array:
        """
        Convert an integer to a binary array of size num_cells.
        
        Parameters:
        action (int): The integer to be converted.
        num_cells (int): The size of the resulting binary array.
        
        Returns:
        np.array: The binary array representation of the integer.
        """

        assert action >= 0, f'Action must be non-negative, got {action}.'
        assert action < 2**self.num_cells, f'Action must be less than 2^{self.num_cells}, got {action}.'

        binary_str = format(action, f'0{self.num_cells}b')  # Convert to binary string with leading zeros
        return np.array([int(bit) for bit in binary_str], dtype=np.int8)





    def discharge(self, action: int) -> None:
        """
        Discharge the battery cells by a constant current.
        
        Parameters:
        action (int): The action taken
        """

        state_SoC = self.map_voltage_to_soc(self.state, self.k_tanh_params)
        switch_action = self.int_action_to_switch_action(action)


        state_SoC = state_SoC - (self.I_CURRENT * self.TIMESTEP / self.Q_cells) * switch_action

        self.state_soc = state_SoC

        self.state = self.map_soc_to_voltage(state_SoC, self.k_tanh_params)


    def step(self, action) -> tuple:
        """
        Execute one time step within the environment.
        
        Parameters:
        action (int): An action provided by the environment
        
        Returns:
        state (np.array): The next state of the environment
        reward (float): The reward for the current action
        done (bool): Whether the episode has ended
        info (dict): Additional information about the environment
        """

        state = self.get_state().copy()
        self.discharge(action)
        state_next = self.get_state().copy()
        reward = self.get_reward(state, state_next, action)


        done = bool(self.is_done())
        truncated = bool(done)
        info = {}
        return state_next, reward, done, truncated, info
    

    def get_reward(self, state: np.array, state_next: np.array, action: int) -> float:
        """
        Calculate and return the reward for a given action.
        
        The reward is based on the reduction in the standard deviation of the state of charge (SoC)
        across the battery cells. A lower standard deviation in the next state indicates a more balanced
        SoC, which is desirable.
        
        Parameters:
        state (np.array): The state of the environment before the action.
        state_next (np.array): The state of the environment after the action.
        w (float): A weight factor to scale the reward. Default is 1.0.
        
        Returns:
        float: The calculated reward.
        """
        reward =  (np.std(state) -  np.std(state_next))* self.w_reward 


        if action == 0:
            reward = -100


        return reward
    

    def is_done(self) -> bool:
        """
        Determine if the episode is done.
        
        Returns:
        done (bool): Whether the episode has ended
        """
        return self.state.min() <= (self.MIN_VOLTAGE + 1e-5)
    

    
    def reset(self, seed=None):
        """
        Reset the state of the environment to an initial state.
        
        Parameters:
        seed (int): Seed for random number generator.
        
        Returns:
        state (np.array): The initial state of the environment
        info (dict): Additional info
        """

        if seed is not None:
            np.random.seed(seed)

        self._initialize_state()
        state = self.get_state()
        info = {}  # Add any additional info you want to return


        return state, info

    def render(self, mode='human', close=False) -> None:
        """
        Render the environment to the screen.
        
        Parameters:
        mode (str): The mode to render with
        close (bool): Whether to close the rendering window
        """
        pass

    def __str__(self) -> str:
        return (f"BMSenv(num_cells={self.num_cells},\n"
                f"        k_tanh_params={self.k_tanh_params},\n"
                f"        Q_cells={self.Q_cells},\n"
                f"        MAX_VOLTAGE={self.MAX_VOLTAGE},\n"
                f"        MIN_VOLTAGE={self.MIN_VOLTAGE},\n"
                f"        INIT_SOC={self.INIT_SOC},\n"
                f"        I_CURRENT={self.I_CURRENT},\n"
                f"        TIMESTEP={self.TIMESTEP},\n"
                f"        w_reward={self.w_reward},\n"
                f"        current_state={self.state},\n"
                f"        current_soc={self.state_soc})")


    

    

    