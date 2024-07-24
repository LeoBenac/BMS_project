import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BMSenv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    

    MAX_VOLTAGE = 4.1
    MIN_VOLTAGE = 3.3
    MAX_SOC = 0.9
    MIN_SOC = 0.1
    RESISTOR_SHUNT = 6
    TIMESTEP = 1e-1
    INIT_SOC_MAX = 0.3
    INIT_SOC_MIN = 0.1
    I_CURRENT = -2.35
    R0 = 0.106033
    R1 = 0.0322
    C1 = 921.61
    R2 = 0.008266
    C2 = 730.31
    Q_default = 3_400
    k_default = 0.0
    R_SHUNT = 6





    
    def __init__(self, num_cells: int = 2,  k_tanh_params : list = [k_default, k_default],
                   Q_cells: list = [Q_default, Q_default], w_reward= 100.0):
        
        super(BMSenv, self).__init__()

        # Define action and observation space
        self.observation_space = spaces.Box(low= self.MIN_SOC, high=self.MAX_SOC , shape=(num_cells,), dtype=np.float64)
        self.action_space = spaces.MultiBinary(num_cells)
        self.num_cells = num_cells

        assert len(k_tanh_params) == self.num_cells, "Size of k_tanh_params must match num_cells"
        assert len(Q_cells) == self.num_cells, "Size of Q_cells must match num_cells"

        self.k_tanh_params = np.array(k_tanh_params)
        self.Q_cells = np.array(Q_cells)
       
        # self._initialize_state()
        self.w_reward = w_reward


    def _initialize_state(self) -> None:
        self.state = np.random.uniform(self.INIT_SOC_MIN, self.INIT_SOC_MAX, self.num_cells)
        self.state = np.random.uniform(0.1, 0.1, self.num_cells)
        self.state_voltage = self.map_soc_to_voltage(self.state, self.k_tanh_params)
        self.i_R1 = 0
        self.i_R2 = 0

    def get_state(self) -> np.array:
        return self.state
    
    
    def map_soc_to_voltage_non_adjusted(self, soc):
        soc_min = self.MIN_SOC
        soc_max = self.MAX_SOC
        voltage_min = self.MIN_VOLTAGE
        voltage_max = self.MAX_VOLTAGE
        
        scale = (voltage_max - voltage_min) / 2
        shift = (voltage_max + voltage_min) / 2
        stretch = np.arctanh(0.9) / (soc_max - (soc_max + soc_min) / 2)
        
        return scale * np.tanh(stretch * (soc - (soc_max + soc_min) / 2)) + shift

    def map_soc_to_voltage(self, soc, k):       
        base_voltage = self.map_soc_to_voltage_non_adjusted(soc)
        soc_min = self.MIN_SOC
        soc_max = self.MAX_SOC
        
        # Apply a consistent vertical shift that is zero at the endpoints
        vertical_shift = k * (1 - np.cos(np.pi * (soc - soc_min) / (soc_max - soc_min)))
        
        return base_voltage - vertical_shift
    


    def charge(self, action: int) -> None:
        """
        Discharge the battery cells by a constant current.
        
        Parameters:
        action (int): The action taken


        """

        if type(action) == list or type(action) == tuple:
            action = np.array(action)

        # state_SoC = self.map_voltage_to_soc(self.state, self.k_tanh_params)
        state = self.state
        # switch_action = self.int_action_to_switch_action(action)
        switch_action = action.copy()

        self.i_R1 = self.i_R1 +  (-1  + np.exp(-1/(self.R1 * self.C1)))*self.i_R1 + (1 - np.exp(-1/(self.R1 * self.C1)))*self.I_CURRENT 
        self.i_R2 = self.i_R2 +  (-1  + np.exp(-1/(self.R2 * self.C2)))*self.i_R2 + (1 - np.exp(-1/(self.R2 * self.C2)))*self.I_CURRENT

        self.state_voltage = self.map_soc_to_voltage(state, self.k_tanh_params) - self.R0 * self.I_CURRENT - self.i_R1 * self.R1 - self.i_R2 * self.R2


        next_state = state - ( ( self.I_CURRENT + (switch_action * (self.state_voltage/self.R_SHUNT))  ) * (self.TIMESTEP / self.Q_cells) )     

        self.state = next_state




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
        if type(action) == list or type(action) == tuple:
            action = np.array(action)

        state = self.get_state().copy()
        state_voltage = self.state_voltage.copy()

        self.charge(action)

        state_next = self.get_state().copy()
        state_voltage_next = self.state_voltage.copy()

        reward = self.get_reward(state, state_next, action, state_voltage, state_voltage_next)


        done = bool(self.is_done())
        truncated = bool(done)
        info = {}
        
        return state_next, reward, done, truncated, info
    

    def get_reward(self, state: np.array, state_next: np.array, action: int, state_soc: np.array, state_soc_next: np.array) -> float:
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
        

        reward =  (np.std(state) -  np.std(state_next))* self.w_reward  - (np.max(state_next) - np.min(state))/(self.w_reward )




        return reward
    

    def is_done(self) -> bool:
        """
        Determine if the episode is done.
        
        Returns:
        done (bool): Whether the episode has ended
        """
        return self.state.max() >= (self.MAX_SOC - 1e-6)
    

    
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
                f"        INIT_SOC_MAX={self.INIT_SOC_MAX},\n"
                f"        INIT_SOC_MIN={self.INIT_SOC_MIN},\n"
                f"        I_CURRENT={self.I_CURRENT},\n"
                f"        TIMESTEP={self.TIMESTEP},\n"
                f"        w_reward={self.w_reward},\n"
                f"        current_state={self.state},\n"
                f"        current_soc={self.state_soc})")


    

    

    