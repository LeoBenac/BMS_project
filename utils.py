import numpy as np
import matplotlib.pyplot as plt

def plot_bms_evolution(bms, states, states_soc, actions, rewards, dones):
    colors = plt.cm.tab10.colors[:bms.num_cells]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plotting the evolution of states for each cell
    for i in range(bms.num_cells):
      start_idx = 0
      for j in range(len(states) -1):
          if actions[j][i] != actions[start_idx][i]:
              linestyle = '-' if actions[start_idx][i] == 1 else '--'
              axs[0, 0].plot(range(start_idx, j+1), [states[k][i] for k in range(start_idx, j+1)], 
                            color=colors[i], linestyle=linestyle, label=f'Cell {i+1}' if start_idx == 0 else "")
              start_idx = j
      # Plot the last segment
      linestyle = '-' if actions[start_idx][i] == 1 else '--'
      axs[0, 0].plot(range(start_idx, len(states)), [states[k][i] for k in range(start_idx, len(states))], 
                    color=colors[i], linestyle=linestyle, label=f'Cell {i+1}' if start_idx == 0 else "") 
      
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('State')
    axs[0, 0].set_title('State vs Time Step for Each Cell')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(bms.MIN_VOLTAGE-0.5, bms.MAX_VOLTAGE+0.5)


    # Plotting the evolution of states_soc for each cell
    for i in range(bms.num_cells):
        start_idx = 0
        for j in range(len(states_soc) - 1):
            if actions[j][i] != actions[start_idx][i]:
                linestyle = '-' if actions[start_idx][i] == 1 else '--'
                axs[0, 1].plot(range(start_idx, j+1), [states_soc[k][i] for k in range(start_idx, j+1)], 
                              color=colors[i], linestyle=linestyle, label=f'Cell {i+1}' if start_idx == 0 else "")
                start_idx = j
        # Plot the last segment
        linestyle = '-' if actions[start_idx][i] == 1 else '--'
        axs[0, 1].plot(range(start_idx, len(states_soc)), [states_soc[k][i] for k in range(start_idx, len(states_soc))], 
                      color=colors[i], linestyle=linestyle, label=f'Cell {i+1}' if start_idx == 0 else "") 

    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('State of Charge (SOC)')
    axs[0, 1].set_title('State of Charge (SOC) vs Time Step for Each Cell')
    axs[0, 1].legend()
    axs[0, 1].set_ylim((0, 1))


    # Plotting the evolution of actions for each cell
    for i in range(bms.num_cells):
        axs[1, 0].scatter( np.arange(1, len(actions) + 1),  [actions[j][i] for j in range(len(actions))], color=colors[i], label=f'Cell {i+1}')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Action')
    axs[1, 0].set_title('Action vs Time Step for Each Cell')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-0.5, 1.5)

    # Plotting the evolution of rewards
    axs[1, 1].scatter(np.arange(1, len(rewards) +1) , rewards)
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Reward')
    axs[1, 1].set_title('Reward vs Time Step')

    plt.tight_layout()
    plt.show()



def plot_voltage_vs_soc(bms):
    """
    Plots the Voltage vs State of Charge (SOC) for each cell in the BMS environment.
    
    Parameters:
    bms (BMSenv): The BMS environment instance.
    """
    # Generate a range of voltage values
    voltages = np.linspace(bms.MIN_VOLTAGE, bms.MAX_VOLTAGE, 100)
    
    # Compute SOC values for each cell using the map_voltage_to_soc method
    soc_values = np.array([bms.map_voltage_to_soc(voltages, k) for k in bms.k_tanh_params])
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    for i in range(bms.num_cells):
        plt.plot(voltages, soc_values[i], label=f'Cell {i+1}')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('State of Charge (SOC)')
    plt.title('Voltage vs State of Charge (SOC) for Each Cell')
    plt.legend()
    plt.show()



def discretize(value: float, bins: np.array) -> int:
    """
    Discretize a continuous value into a bin number based on predefined ranges and densities.

    Parameters:
    value (float): The continuous value to be discretized.

    Returns:
    int: The bin number corresponding to the input value.

    Raises:
    ValueError: If the value is outside the allowed range.
    """
    if not (2.2 <= value < 4.2):
        raise ValueError("Value must be between 2.2 and 4.2")

    bin_number = np.digitize(value, bins) - 1  # Subtract 1 to get 0-based index
    return bin_number

def discretize_features(values: np.array, bins: np.array) -> np.array:
    """
    Discretize multiple continuous features into bin numbers based on predefined ranges and densities.

    Parameters:
    values (np.array): The continuous values to be discretized.

    Returns:
    np.array: The bin numbers corresponding to the input values.
    """
    return np.array([discretize(value, bins) for value in values])

def combination_to_integer(discretized_values: np.array, num_bins: int) -> int:
    """
    Convert a combination of discretized values to a unique integer.

    Parameters:
    discretized_values (np.array): The array of discretized values.
    num_bins (int): The number of bins used for discretization.

    Returns:
    int: A unique integer representing the combination of discretized values.
    """
    unique_integer = 0
    for i, value in enumerate(discretized_values):
        unique_integer += value * (num_bins ** i)
    return unique_integer

def features_to_unique_integer(features: np.array, bins: np.array) -> int:
    """
    Convert features to a unique integer based on discretization.

    Parameters:
    features (np.array): The continuous features to be discretized.
    bins (np.array): The bins used for discretization.

    Returns:
    int: A unique integer representing the combination of discretized features.
    """
    discretized_features = discretize_features(features, bins)
    num_bins = len(bins) - 1  # Corrected number of bins
    return combination_to_integer(discretized_features, num_bins)
