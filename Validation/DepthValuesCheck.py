import numpy as np
import matplotlib.pyplot as plt

# Read the txt file
file_path = 'Validation\Depth_data\depth_values.txt'

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

plotrange_x = range(0, max(meter_range)+1)
plotrange_y = range(int(min_y), int(max_y)+1)
# Plotting 'Real Depth vs Measured/Adjusted depth'
plt.plot(meter_range, mean_measured, label='Measured', marker='o', linestyle='-', color='b')
plt.plot(meter_range, mean_adjusted, label='Adjusted', marker='o', linestyle='-', color='r')
plt.plot([0, 10], [0, 10], linestyle=':', color='black', label='Ideal Depth')
plt.legend()
plt.grid(True)

# Plotting 'Deviation'
plt.plot(meter_range, error_measured, marker='o', linestyle='-', color='b')
plt.plot(meter_range, error_adjusted, marker='o', linestyle='-', color='r')
plt.axhline(0, color='black', linestyle='--', label='Ideal Error')
plt.xlabel('Real Depth [m]')
plt.ylabel('Measured/Adjusted Depth & Deviation [m]')
plt.title('Real Depth vs Measured/Adjusted Depth')
plt.legend()
plt.grid(True)
plt.axhspan(
    ymin=min(min(error_measured), min(error_adjusted)),
    ymax=max(max(error_measured), max(error_adjusted)),
    color='red', alpha=0.1
)
plt.text(
    x=1.5,
    y=-2.3,
    s="Error Range",
    fontsize=12,
    color='red',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)
plt.axhspan(
    ymin = 0,
    ymax=max(max(mean_measured), max(mean_adjusted)),
    color='green', alpha=0.1
)
plt.text(
    x=6.7,
    y=3.7,
    s="Depth Range",
    fontsize=12,
    color='green',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='green')
)
plt.yticks(plotrange_y), plt.xticks(plotrange_x)
plt.savefig("Validation/figures/RealvsMeasuredDepth")
plt.show()
