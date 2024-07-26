import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class BMSenv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    

    MAX_VOLTAGE = 4.1
    MIN_VOLTAGE = 3.3
    MAX_SOC = 0.9
    MIN_SOC = 0.1
    TIMESTEP = 30/3600
    INIT_SOC_MAX = 0.3
    INIT_SOC_MIN = 0.1
    I_CURRENT = -2.35
    Q_default = 3.4
    k_default = 0.0
    R_SHUNT = 4
    DATA = np.array([
    [0.9, 0.1063, 0.0303, 726.32, 0.0099, 636.78],
    [0.8, 0.1016, 0.0302, 734.10, 0.0102, 594.41],
    [0.7, 0.1020, 0.0315, 766.53, 0.0105, 575.44],
    [0.6, 0.1023, 0.0390, 929.76, 0.0078, 613.48],
    [0.5, 0.1024, 0.0271, 1131.40, 0.0078, 789.73],
    [0.4, 0.1040, 0.0275, 1161.67, 0.0077, 791.77],
    [0.3, 0.1042, 0.0272, 1128.74, 0.0077, 788.39],
    [0.2, 0.1070, 0.0272, 982.50, 0.0076, 746.24],
    [0.1, 0.1325, 0.0498, 747.54, 0.0096, 639.61]
        ])
    models = []





    
    def __init__(self, num_cells: int = 2,  k_tanh_params : list = [k_default, k_default],
                   Q_cells: list = [Q_default, Q_default], w_reward= 100.0):
        
        super(BMSenv, self).__init__()

        # Define action and observation space
        low = np.array([self.MIN_SOC] * num_cells + [self.MIN_VOLTAGE - 1] * num_cells)
        high = np.array([self.MAX_SOC] * num_cells + [self.MAX_VOLTAGE + 1] * num_cells)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.MultiBinary(num_cells)
        self.num_cells = num_cells

        assert len(k_tanh_params) == self.num_cells, "Size of k_tanh_params must match num_cells"
        assert len(Q_cells) == self.num_cells, "Size of Q_cells must match num_cells"

        self.k_tanh_params = np.array(k_tanh_params)
        self.Q_cells = np.array(Q_cells)
       
        # self._initialize_state()
        self.w_reward = w_reward

        data = self.DATA    


        # Splitting the data
        X = data[:, 0].reshape(-1, 1)  # SoC values
        Y = data[:, 1:]  # Remaining columns

        # Transform input data to include quadratic terms
        poly = PolynomialFeatures(degree= 4)
        X_poly = poly.fit_transform(X)

        # Train a model for each output
        for i in range(Y.shape[1]):
            model = LinearRegression()
            model.fit(X_poly, Y[:, i])
            self.models.append(model)

        self.poly = poly


    def _initialize_state(self, seed = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        init_state_soc = np.random.uniform(self.INIT_SOC_MIN, self.INIT_SOC_MAX, self.num_cells)
        # init_state_soc = np.random.uniform(0.1, 0.1, self.num_cells)
        init_state_voltage = self.map_soc_to_voltage(init_state_soc, self.k_tanh_params)
        self.state = np.concatenate([init_state_soc, init_state_voltage])
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
        state_soc = state[:self.num_cells].copy()
        state_voltage = state[self.num_cells:].copy()

        R0 = self.models[0].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        R1 = self.models[1].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        C1 = self.models[2].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        R2 = self.models[3].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        C2 = self.models[4].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))

        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2

        # print(f"R0: {R0}, R1: {R1}, C1: {C1}, R2: {R2}, C2: {C2}")


        switch_action = action.copy()

        self.i_R1 = self.i_R1 +  (-1  + np.exp(-1/(self.R1 * self.C1)))*self.i_R1 + (1 - np.exp(-1/(self.R1 * self.C1)))*self.I_CURRENT 
        self.i_R2 = self.i_R2 +  (-1  + np.exp(-1/(self.R2 * self.C2)))*self.i_R2 + (1 - np.exp(-1/(self.R2 * self.C2)))*self.I_CURRENT

        state_voltage_next = self.map_soc_to_voltage(state_soc, self.k_tanh_params) - self.R0 * (self.I_CURRENT  ) - self.i_R1 * self.R1 - self.i_R2 * self.R2

        # print('\nstate_voltage_next: ', state_voltage_next)

        # state_voltage_next = self.map_soc_to_voltage(state_soc, self.k_tanh_params) - self.R0 * (self.I_CURRENT + (switch_action * (state_voltage/self.R_SHUNT)) ) - self.i_R1 * self.R1 - self.i_R2 * self.R2
        state_soc_next = state_soc - ( ( self.I_CURRENT + (switch_action * (state_voltage/self.R_SHUNT))  ) * (self.TIMESTEP / self.Q_cells) )     


        self.state = np.concatenate([state_soc_next, state_voltage_next])

    def discharge(self) -> None:
        """
        Discharge the battery cells by a constant current.
        """
        state = self.state
        state_soc = state[:self.num_cells].copy()
        state_voltage = state[self.num_cells:].copy()

        R0 = self.models[0].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        R1 = self.models[1].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        C1 = self.models[2].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        R2 = self.models[3].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))
        C2 = self.models[4].predict(self.poly.fit_transform(state_soc.reshape(-1, 1)))

        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2

        self.i_R1 = self.i_R1 +  (-1  + np.exp(-1/(self.R1 * self.C1)))*self.i_R1 + (1 - np.exp(-1/(self.R1 * self.C1)))* (-self.I_CURRENT)
        self.i_R2 = self.i_R2 +  (-1  + np.exp(-1/(self.R2 * self.C2)))*self.i_R2 + (1 - np.exp(-1/(self.R2 * self.C2)))* (-self.I_CURRENT)

        state_voltage_next = self.map_soc_to_voltage(state_soc, self.k_tanh_params) - self.R0 * ((-self.I_CURRENT)  ) - self.i_R1 * self.R1 - self.i_R2 * self.R2
        state_soc_next = state_soc - ( ( (-self.I_CURRENT) ) * (self.TIMESTEP / self.Q_cells) )     

        self.state = np.concatenate([state_soc_next, state_voltage_next])


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

        state_soc = self.state[:self.num_cells].copy()
        state_voltage = self.state[self.num_cells:].copy()

        self.charge(action)

        state_next = self.get_state().copy()

        state_soc_next = state_next[:self.num_cells].copy()
        state_voltage_next = state_next[self.num_cells:].copy

        reward = self.get_reward(state, action, state_next)


        done = bool(self.is_done())
        truncated = bool(done)
        info = {}
        
        return state_next, reward, done, truncated, info
    

    def get_reward(self, state: np.array,  action: np.array, state_next: np.array) -> float:
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

        state_soc = state[:self.num_cells].copy()
        state_voltage = state[self.num_cells:].copy()

        state_soc_next = state_next[:self.num_cells].copy()
        state_voltage_next = state_next[self.num_cells:].copy()


        

        reward =  ((np.std(state_soc) -  np.std(state_soc_next))* self.w_reward  - (np.max(state_soc_next) - np.min(state_soc))/(self.w_reward ) ) \
                # + ((np.std(state_voltage) -  np.std(state_voltage_next))* self.w_reward  - (np.max(state_voltage_next) - np.min(state_voltage))/(self.w_reward ) )




        return reward
    

    def is_done(self) -> bool:
        """
        Determine if the episode is done.
        
        Returns:
        done (bool): Whether the episode has ended
        """

        state = self.get_state()
        state_soc = state[:self.num_cells].copy()

        return state_soc.max() >= (self.MAX_SOC - 1e-6)
    

    
    def reset(self, seed=None):
        """
        Reset the state of the environment to an initial state.
        
        Parameters:
        seed (int): Seed for random number generator.
        
        Returns:
        state (np.array): The initial state of the environment
        info (dict): Additional info
        """

        

        self._initialize_state(seed=seed)
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
                f"        SOC={self.state[:self.num_cells]},\n"
                f"        VOLTAGE={self.state[self.num_cells:]},\n"
                f"        w_reward={self.w_reward},\n"
                f"        R_SHUNT={self.R_SHUNT}"
                f"        R0={self.R0}"
                f"        R1={self.R1}"
                f"        C1={self.C1}"
                f"        R2={self.R2}"
                f"        C2={self.C2}")



    

    

    