import numpy as np
import matplotlib.pyplot as plt

def plot_bms_evolution(bms, states, states_soc, actions, rewards, dones):
    colors = plt.cm.tab10.colors[:bms.num_cells]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plotting the evolution of states for each cell
    for i in range(bms.num_cells):
        for j in range(len(states) - 1):
            axs[0, 0].plot([j, j+1], [states[j][i], states[j+1][i]], color=colors[i], label=f'Cell {i+1}' if j == 0 else "")
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('State')
    axs[0, 0].set_title('State vs Time Step for Each Cell')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(bms.MIN_VOLTAGE-0.5, bms.MAX_VOLTAGE+0.5)

    # Plotting the evolution of states_soc for each cell
    for i in range(bms.num_cells):
        for j in range(len(states_soc) - 1):
            axs[0, 1].plot([j, j+1], [states_soc[j][i], states_soc[j+1][i]], color=colors[i], label=f'Cell {i+1}' if j == 0 else "")
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('State of Charge (SOC)')
    axs[0, 1].set_title('State of Charge (SOC) vs Time Step for Each Cell')
    axs[0, 1].legend()
    axs[0, 1].set_ylim((0, 1))

    # Plotting the evolution of actions for each cell
    for i in range(bms.num_cells):
        axs[1, 0].plot([actions[j][i] for j in range(len(actions))], color=colors[i], label=f'Cell {i+1}')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Action')
    axs[1, 0].set_title('Action vs Time Step for Each Cell')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(-0.5, 1.5)

    # Plotting the evolution of rewards
    axs[1, 1].plot(rewards)
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

