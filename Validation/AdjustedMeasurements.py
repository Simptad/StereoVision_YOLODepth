import numpy as np
import matplotlib.pyplot as plt

# Read the txt file
file_path = 'Validation/Depth_data/depth_values.txt'

meter_range = range(1, 11)

# Initialize data structures
meters_data = {i: {'measured': [], 'adjusted': []} for i in meter_range}

# Read the file and process the data
with open(file_path, 'r') as f:
    lines = f.readlines()

    current_meter = None
    for line in lines:
        line = line.strip()
        
        # Check if the line has the meter title (e.g., (1) meter)
        if line.startswith('(') and 'meter' in line:
            current_meter = int(line.split(')')[0][1:])
        
        # Check if the line contains measured or adjusted values
        elif line.startswith('measured') or line.startswith('adjusted'):
            # Extract the value adjusted 'measured: ' or 'adjusted: '
            value = float(line.split(':')[1].strip())
            if line.startswith('measured'):
                meters_data[current_meter]['measured'].append(value)
            elif line.startswith('adjusted'):
                meters_data[current_meter]['adjusted'].append(value)


mean_measured = []; mean_adjusted = []; error_measured = []; error_adjusted = []
for meter in meter_range:
    mean_meas = np.mean(meters_data[meter]['measured'])
    mean_adj = np.mean(meters_data[meter]['adjusted'])

    mean_measured.append(mean_meas)
    mean_adjusted.append(mean_adj)
    error_measured.append(meter-mean_meas)
    error_adjusted.append(meter-mean_adj)

min_y = min(min(mean_measured), min(mean_adjusted), min(error_measured), min(error_adjusted))
max_y = max(max(mean_measured), max(mean_adjusted), max(error_measured), max(error_adjusted))

plotrange_x = range(0, max(meter_range) + 1)
plotrange_y = range(0, int(max_y) + 1)  # Only positive ticks for display

# Use tab10 color scheme
colors = plt.cm.tab10.colors

plt.figure(figsize=(12, 8))

# Plot 'Real Depth vs Measured/Adjusted Depth'
plt.legend(title="Scaling factor: 1.9")
plt.plot(meter_range, mean_measured, label='Measured', marker='o', linestyle='-', color=colors[0])
plt.plot(meter_range, mean_adjusted, label='Adjusted', marker='o', linestyle='-', color=colors[1])
# plt.plot([0, 15], [0, 15], linestyle='--', color="black", label='Ideal Depth')  # Extend the line to 15

# Plot 'Deviation' (unchanged)
plt.plot(meter_range, error_measured, marker='o', linestyle='-', color=colors[0])
plt.plot(meter_range, error_adjusted, marker='o', linestyle='-', color=colors[1])

plt.axhline(0, color="black", linestyle='--', label='Ideal Deviation')

plt.xlabel('True Depth [m]', fontsize=18)
plt.ylabel('Measured/Adjusted Depth & Deviation [m]', fontsize=18)
plt.title('True Depth vs Measured/Adjusted Depth', fontsize=18)

plt.legend(title="Scaling factor: 1.9", title_fontsize=18, fontsize=18)
plt.grid(True)

# Highlight error range
plt.axhspan(
    ymin=min(error_measured + error_adjusted)-1,
    ymax=max(error_measured + error_adjusted),
    color=colors[3], alpha=0.1
)
plt.text(
    x=1.5,
    y=-2.3,
    s="Deviation Range",
    fontsize=12,
    color=colors[3],
    bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors[3])
)

# Highlight depth range
plt.axhspan(
    ymin=0,
    ymax=max(max(mean_measured), max(mean_adjusted))+1,
    color="green", alpha=0.1
)
plt.text(
    x=6.7,
    y=3.7,
    s="Depth Range",
    fontsize=12,
    color="green",
    bbox=dict(facecolor='white', alpha=0.7, edgecolor="green")
)

plt.xticks(plotrange_x)
# Adjust the y-axis to mirror negative values visually
plt.ylim(min_y-1, max_y+1)  # Ensure full range is shown
plt.yticks(
    range(int(min_y), int(max_y) + 1),
    [abs(y) for y in range(int(min_y), int(max_y) + 1)]  # Display absolute values
)

# Set fixed x-axis and y-axis limits
plt.xlim(0, 11)  # Keep the x-axis range fixed to the current visible area
plt.ylim(min_y - 1, max_y + 1)  # Keep the y-axis range fixed to the current visible area

# Plot 'Real Depth vs Measured/Adjusted Depth'
plt.plot(meter_range, mean_measured, label='Measured', marker='o', linestyle='-', color=colors[0])
plt.plot(meter_range, mean_adjusted, label='Adjusted', marker='o', linestyle='-', color=colors[1])
plt.plot([0, 15], [0, 15], linestyle='--', color="black", label='Ideal Depth')  # Extend the line to 15

plt.tight_layout()  # Automatically adjust subplot parameters to fit the figure

plt.savefig("Validation/figures/AdjustedMeasurements.png")
plt.show()