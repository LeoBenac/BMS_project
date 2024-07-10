import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_bms_evolution(bms, states, states_soc, actions, rewards, dones, include_bad_rewards=False):
    
    colors = plt.cm.tab10.colors[:bms.num_cells]

    colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
    'blue', 'green', 'red', 'coral', 'navy', 'lime', 'teal', 'violet', 'gold', 'indigo', 'turquoise',
    'darkgreen', 'salmon', 'chocolate', 'maroon', 'orchid', 'plum', 'sienna', 'tan', 'crimson', 'darkblue'
    ]

    # Ensure colors has at least `bms.num_cells` elements, otherwise cycle through available colors
    if len(colors) < bms.num_cells:
        colors = colors * (bms.num_cells // len(colors) + 1)
    else:
        colors = colors[:bms.num_cells]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plotting the evolution of states for each cell
    for i in range(bms.num_cells):
        axs[0, 0].plot(range(len(states)), [states[k][i] for k in range(len(states))],
                    color=colors[i], label=f'Cell {i + 1}')
        
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('Voltage')
    axs[0, 0].set_title('Voltage vs Time Step for Each Cell')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(bms.MIN_VOLTAGE - 0.5, bms.MAX_VOLTAGE + 0.5)

    # Plotting the evolution of states_soc for each cell
        # Plotting the evolution of states_soc for each cell
    for i in range(bms.num_cells):
        axs[0, 1].plot(range(len(states_soc)), [states_soc[k][i] for k in range(len(states_soc))],
                       color=colors[i], label=f'Cell {i + 1}')
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('State of Charge (SOC)')
    axs[0, 1].set_title('State of Charge (SOC) vs Time Step for Each Cell')
    axs[0, 1].legend()
    axs[0, 1].set_ylim((0, 1))

    # Plotting the evolution of actions for each cell
    for i in range(bms.num_cells):
        for j in range(len(actions)):
            color = 'blue' if actions[j][i] == 1 else 'red'
            axs[1, 0].barh(i, 1, left=j, color=color, edgecolor='none', height=0.8)
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Cell Number')
    axs[1, 0].set_title('Action vs Time Step for Each Cell')
    axs[1, 0].set_yticks(range(bms.num_cells))  # Fix y-ticks
    axs[1, 0].set_yticklabels([f'Cell {i + 1}' for i in range(bms.num_cells)])  # Add labels to y-ticks
    axs[1, 0].set_ylim(-0.5, bms.num_cells - 0.5)
    axs[1, 0].set_xlim(0, len(actions))

    # Add legend for colors
    legend_elements = [Line2D([0], [0], color='blue', lw=4, label='ON'),
                       Line2D([0], [0], color='red', lw=4, label='OFF')]
    axs[1, 0].legend(handles=legend_elements, loc='upper right')

    # Plotting the evolution of accumulated rewards
    if not include_bad_rewards:
        rewards = [np.nan if reward == -100 else reward for reward in rewards]

    # Calculate the accumulated rewards
    accumulated_rewards = np.nancumsum(rewards)  # Use np.nancumsum to handle NaN values

    # Normalize the accumulated rewards by the time step
    time_steps = np.arange(1, len(accumulated_rewards) + 1)
    normalized_accumulated_rewards = accumulated_rewards / time_steps

    # Plot the normalized accumulated rewards
    axs[1, 1].plot(time_steps, normalized_accumulated_rewards, label='Normalized Accumulated Rewards')
    axs[1, 1].set_xlabel('Time Step')
    axs[1, 1].set_ylabel('Normalized Accumulated Rewards')
    axs[1, 1].set_title('Normalized Accumulated Rewards vs Time Step')
    axs[1, 1].legend()

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
    if not (0.1 <= value < 0.9):
        raise ValueError("Value must be between 0.1 and 0.9")

    bin_number = np.digitize(value, bins) - 1  # Subtract 1 to get 0-based index
    return bin_number

# def discretize_features(values: np.array, bins: np.array) -> np.array:
#     """
#     Discretize multiple continuous features into bin numbers based on predefined ranges and densities.

#     Parameters:
#     values (np.array): The continuous values to be discretized.

#     Returns:
#     np.array: The bin numbers corresponding to the input values.
#     """
#     return np.array([discretize(value, bins) for value in values])


def discretize_features(values: np.array, bins: np.array) -> np.array:
    """
    Discretize multiple continuous features into bin numbers based on predefined ranges and densities.

    Parameters:
    values (np.array): The continuous values to be discretized.
    bins (np.array): The bins used for discretization.

    Returns:
    np.array: The bin numbers corresponding to the input values.

    """
    values = np.array(values)


    if not np.all((0.1 <= values) & (values < 0.9)):
        raise ValueError("All values must be between 0.1 and 0.9")

    # Use np.digitize directly on the entire array
    bin_numbers = np.digitize(values, bins) - 1  # Subtract 1 to get 0-based index
    return bin_numbers

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



def integer_to_combination(unique_integer: int, num_bins: int, num_dimensions: int) -> np.array:
    """
    Convert a unique integer back to a combination of discretized values.

    Parameters:
    unique_integer (int): The unique integer representing the combination of discretized values.
    num_bins (int): The number of bins used for discretization.
    num_dimensions (int): The number of dimensions (features).

    Returns:
    np.array: The array of discretized values.
    """
    discretized_values = np.zeros(num_dimensions, dtype=int)
    for i in range(num_dimensions):
        discretized_values[i] = unique_integer % num_bins
        unique_integer //= num_bins
    return discretized_values

def bin_boundaries(discretized_value: int, bins: np.array) -> tuple:
    """
    Get the bin boundaries for a given discretized value.

    Parameters:
    discretized_value (int): The discretized bin number.
    bins (np.array): The bins used for discretization.

    Returns:
    tuple: The lower and upper boundaries of the bin.
    """
    return bins[discretized_value], bins[discretized_value + 1]

def unique_integer_to_bin_boundaries(unique_integer: int, bins: np.array, num_dimensions: int) -> list:
    """
    Convert a unique integer to the bin boundaries for each dimension.

    Parameters:
    unique_integer (int): The unique integer representing the combination of discretized values.
    bins (np.array): The bins used for discretization.
    num_dimensions (int): The number of dimensions (features).

    Returns:
    list: A list of tuples representing the bin boundaries for each dimension.
    """
    num_bins = len(bins) - 1  # Corrected number of bins
    discretized_values = integer_to_combination(unique_integer, num_bins, num_dimensions)
    return [bin_boundaries(value, bins) for value in discretized_values]