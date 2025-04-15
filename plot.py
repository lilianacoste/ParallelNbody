import numpy as np
import matplotlib.pyplot as plt

def read_simulation_output(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split('\t')
            if len(values) < 2:
                continue  # skip empty or bad lines
            num_particles = int(values[0])
            expected_values = 1 + 11 * num_particles
            if len(values) != expected_values:
                print(f"[Warning] Skipping line with unexpected length: {len(values)} vs expected {expected_values}")
                continue
            timestep_data = []
            for i in range(num_particles):
                offset = 1 + i * 11
                particle_data = [float(v) for v in values[offset:offset+11]]
                timestep_data.append(particle_data)
            data.append(timestep_data)
    return np.array(data)

def plot_positions(data, num_steps, num_particles, output_filename="simulation.png"):
    """
    Plots the positions of the particles over the time steps.
    """
    fig, ax = plt.subplots()
    
    for step in range(0, num_steps, 10):  # Change the step interval for plotting frequency
        positions = data[step, :, 1:4]  # Extract positions (x, y, z) for all particles
        ax.scatter(positions[:, 0], positions[:, 1], label=f"Step {step}")
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Particle Positions Over Time')
    ax.legend()
    plt.savefig(output_filename)
    plt.show()

def main():
    # File containing the output from your C++ simulation (e.g., "simulation_output.txt")
    file_path = 'simulation_output.txt'
    num_steps = 100  # Total number of steps to process
    num_particles = 10  # Number of particles in the simulation
    
    # Read the simulation output
    data = read_simulation_output(file_path)
    
    # Reshape the data to have a time axis
    data = data.reshape(num_steps, num_particles, -1)  # Reshape to (steps, particles, values)
    
    # Plot the positions of the particles
    plot_positions(data, num_steps, num_particles)

if __name__ == "__main__":
    main()
