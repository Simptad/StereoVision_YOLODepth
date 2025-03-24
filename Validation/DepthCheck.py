import matplotlib.pyplot as plt

# Filepath to the DepthAccuracy.txt file
file_path = r"c:\Users\SESITAD1\Desktop\StereoVision_YOLODepth\Validation\DepthAccuracy.txt"

# Initialize variables
scaling_factors = []
data = {}

# Read and parse the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    current_scaling_factor = None

    for line in lines:
        line = line.strip()
        if line.startswith("Scaling factor:"):
            # Extract scaling factor
            current_scaling_factor = line.split(":")[1].strip()
            scaling_factors.append(current_scaling_factor)
            data[current_scaling_factor] = {"x": [], "y": []}
        elif line and current_scaling_factor:
            # Extract data points
            x, y = map(float, line.split(","))
            data[current_scaling_factor]["x"].append(x)
            data[current_scaling_factor]["y"].append(y)

# Plot the data
plt.figure(figsize=(10, 6))
for scaling_factor, points in data.items():
    plt.plot(points["x"], points["y"], marker='o', label=scaling_factor)

plt.plot([0, 10], [0, 10], linestyle='--', color='black', label='y = x (reference)')

# Add labels, title, and legend
plt.xlabel("Distance (meters)")
plt.ylabel("Measured Depth (meters)")
plt.title("Depth Accuracy for Different Scaling Factors")
plt.legend(title="Scaling Factors")
plt.grid(True)

# Plot the difference (y - x)
plt.figure(figsize=(10, 6))

# Plot the difference for each scaling factor
for scaling_factor, points in data.items():
    differences = [y - x for x, y in zip(points["x"], points["y"])]  # Calculate y - x for each point
    plt.plot(points["x"], differences, marker='o', label=scaling_factor)

# Add a horizontal line at y = 0 for reference
plt.axhline(0, color='gray', linestyle='--', label='y = 0 (reference)')

# Add labels, title, and legend
plt.xlabel("Distance (meters)")
plt.ylabel("Difference (Measured Depth - Real Distance) (meters)")
plt.title("Difference Between Measured Depth and Real Distance")
plt.legend(title="Scaling Factors")
plt.grid(True)
plt.show()