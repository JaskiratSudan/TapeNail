import time
import psutil
import matplotlib.pyplot as plt

def display_usage(cpu_usage, mem_usage, bars=50):
    """Helper function to format and print usage bars"""
    cpu_percent_val = cpu_usage / 100.0
    cpu_bar = ' ' * int(cpu_percent_val * bars) + '-' * (bars - int(cpu_percent_val * bars))

    mem_percent_val = mem_usage / 100.0
    mem_bar = ' ' * int(mem_percent_val * bars) + '-' * (bars - int(mem_percent_val * bars))

    print(f"\rCPU Usage: |{cpu_bar}| {cpu_usage:.2f}%   ", end="")
    print(f"MEM Usage: |{mem_bar}| {mem_usage:.2f}%   ", end="\r")


def log_and_plot_resources(iterations=200, interval_seconds=1):
    """Monitors resources, logs them to files, and plots the results."""
    cpu_data = []
    mem_data = []
    time_stamps = []
    start_time = time.time()

    print(f"Monitoring CPU and Memory for {iterations * interval_seconds} seconds...")

    for i in range(iterations):
        current_cpu = psutil.cpu_percent()
        current_mem = psutil.virtual_memory().percent
        current_time = time.time() - start_time

        cpu_data.append(current_cpu)
        mem_data.append(current_mem)
        time_stamps.append(current_time)

        display_usage(current_cpu, current_mem) # Display live usage bars
        time.sleep(interval_seconds)

    print("\nMonitoring complete.")

    # --- Logging Data to Files ---
    print("Logging data to files...")
    try:
        with open('cpu_info.txt', 'w') as f_cpu:
            for cpu_val in cpu_data:
                f_cpu.write(f"{cpu_val:.2f}\n")

        with open('mem_info.txt', 'w') as f_mem:
            for mem_val in mem_data:
                f_mem.write(f"{mem_val:.2f}\n")
        print("Data logged successfully.")
    except IOError as e:
        print(f"Error writing to files: {e}")


    # --- Plotting the Results ---
    print("Generating plot...")
    fig, ax1 = plt.subplots(figsize=(12, 6)) # Increased figure size for better label visibility

    # Plot CPU usage
    color_cpu = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CPU Usage (%)', color=color_cpu)
    ax1.plot(time_stamps, cpu_data, color=color_cpu, marker='o', linestyle='-', markersize=4, label='CPU Usage')
    ax1.tick_params(axis='y', labelcolor=color_cpu)
    ax1.set_ylim(0, 105) # Set Y-axis limit for percentage

    # Plot Memory usage on the secondary y-axis
    ax2 = ax1.twinx()
    color_mem = 'tab:blue'
    ax2.set_ylabel('Memory Usage (%)', color=color_mem) # Changed label as we record percent
    ax2.plot(time_stamps, mem_data, color=color_mem, marker='s', linestyle='--', markersize=4, label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor=color_mem)
    ax2.set_ylim(0, 105) # Set Y-axis limit for percentage

    # Adding title and legend
    plt.title('ViKey System Resource Usage Over Time')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes) # Added combined legend
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.grid(True)
    print("Displaying plot...")
    plt.show()
    print("Plot closed.")


# --- Main Execution ---
if __name__ == "__main__":
    # You can adjust the number of iterations and the interval between measurements
    log_and_plot_resources(iterations=60, interval_seconds=1) # This will monitor for 60 seconds, 1 reading per second
