import numpy as np
import matplotlib.pyplot as plt

# Read the txt file
file_path = 'Validation/Depth_data/DepthAccuracy.txt'

# Initialize data structures
scaling_factors = []
data = {}

# Read the file and process the data
with open(file_path, 'r') as f:
    lines = f.readlines()
    current_scaling_factor = None

    for line in lines:
        line = line.strip()
        if line.startswith("Scaling factor:"):
            # Extract scaling factor
            current_scaling_factor = line.split(":")[1].strip()
            scaling_factors.append(current_scaling_factor)
            data[current_scaling_factor] = {"true": [], "measured": []}
        elif line and current_scaling_factor:
            # Extract true and measured values
            try:
                true_value, measured_value = map(float, line.split(","))
                data[current_scaling_factor]["true"].append(true_value)
                data[current_scaling_factor]["measured"].append(measured_value)
            except ValueError:
                print(f"Skipping invalid line: {line}")

# Configuration for x and y axes
x_min, x_max = -1, 11  # Set x-axis range
y_min, y_max = -4, 13  # Set y-axis range
y_ticks = range(y_min, y_max + 1)  # Set y-axis ticks

# Create a combined plot
plt.figure(figsize=(12, 8))

# Define a color map for consistent coloring
colors = plt.cm.tab10(np.linspace(0, 1, len(scaling_factors)))

# Plot 'True Depth vs Measured Depth' and 'Error' for each scaling factor
all_errors = []
all_measured = []
for idx, (scaling_factor, values) in enumerate(data.items()):
    color = colors[idx]  # Assign a unique color for each scaling factor
    # Plot True Depth vs Measured Depth
    plt.plot(
        values["true"], values["measured"],
        marker='o', linestyle='-', color=color, label=f"Scaling {scaling_factor}"
    )
    # Calculate and plot error values (can go below 0)
    errors = [true - measured for true, measured in zip(values["true"], values["measured"])]
    all_errors.extend(errors)
    all_measured.extend(values["measured"])
    plt.plot(
        values["true"], errors,
        marker='x', linestyle='--', color=color
    )

# Add ideal depth line
plt.plot([0, 13], [0, 13], linestyle='--', color='black', label='Ideal Depth')

# Highlight error range
plt.axhspan(
    ymin=min(all_errors) - 1,
    ymax=max(all_errors),
    color='red', alpha=0.1
)
plt.text(
    x=0.3,
    y=-2,
    s="Deviation Range",
    fontsize=12,
    color='red',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)

# Highlight depth range
boxcolor = colors[1]
plt.axhspan(
    ymin=0,
    ymax=max(all_measured) + 1,
    color=boxcolor, alpha=0.1
)
plt.text(
    x=4.4,
    y=8.8,
    s="Depth Range",
    fontsize=12,
    color=boxcolor,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor=boxcolor)
)

# Add a horizontal dotted line at y=0 for "Ideal Error"
plt.axhline(0, color='black', linestyle='--', label='Ideal Deviation')

# Configure the x-axis and y-axis
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Set x-axis and y-axis ticks to exclude the first and last numbers
plt.xticks(range(x_min + 1, x_max))  # Exclude the first and last numbers on the x-axis
plt.yticks(range(y_min + 1, y_max), [abs(y) for y in range(y_min + 1, y_max)])  # Exclude the first and last numbers on the y-axis as absolute values

# Add labels, title, and legend
plt.xlabel("True Depth [m]")
plt.ylabel("Measured Depth & Deviation [m]")
plt.title("True Depth vs Measured Depth for Different Scaling Factors")
plt.legend(title="Scaling Factors (Calibrated between 2-6m)", loc='upper left')
plt.grid(True)

# Save the combined plot
plt.tight_layout()  # Adjust layout to fit legend
plt.savefig("Validation/figures/InitialMeasurements.png")
plt.show()