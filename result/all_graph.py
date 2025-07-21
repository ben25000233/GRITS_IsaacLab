import numpy as np
import json
import matplotlib.pyplot as plt

all_spillage_array = []
all_unspillage_array = []
all_scoop_success_array = []
all_success_array = []
spillage_counts = []

weights = [3, 4, 5, 6, 7.5, 9]

r_radius = 0.009
mass = 0.003
ball_amount = 10



# Initialize arrays to store rates for each weight
metrics = ['Unspillage Rate', 'Scoop Success Rate', 'Success Rate']
rates_by_weight = []

# Load data for each weight
for weight in weights:
    file_name = f"./amount={ball_amount}/R{r_radius}_M{mass}_A{ball_amount}_W{weight}.json"
    with open(file_name, "r") as json_file:
        spillage_data = json.load(json_file)

    all_array = np.array(spillage_data["spillage_scoop"])[:100]

    # Calculate rates
    spillage_array = all_array[:, 0]
    unspillage_rate = 1 - np.count_nonzero(spillage_array) / 100
    spillage_count = np.zeros(8, dtype=int)
    for i in range(1, 9):  # Iterate from 1 to 8
        spillage_count[i - 1] = np.sum(spillage_array == i)
    spillage_counts.append(spillage_count)

    scoop_success_array = all_array[all_array[:, 1] > 0]
    scoop_success_rate = len(scoop_success_array) / len(all_array)

    success_array = scoop_success_array[scoop_success_array[:, 0] == 0]
    success_rate = len(success_array) / len(all_array)

    # Append rates for this weight
    rates_by_weight.append([unspillage_rate, scoop_success_rate, success_rate])

# Convert rates_by_weight to a NumPy array for easier plotting
rates_by_weight = np.array(rates_by_weight)

# --------------Plot the line chart for all --------------------
plt.figure(figsize=(12, 6))
for i, weight in enumerate(weights):
    plt.plot(metrics, rates_by_weight[i], marker='o', linestyle='-', label=f'Weight = {weight}')

plt.title('Rates vs Metrics for Different Weights', fontsize=16)
plt.ylabel('Rate', fontsize=20)
plt.tick_params(axis='x', labelsize=16)  
plt.tick_params(axis='y', labelsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the line chart as an image (optional)
plt.savefig("rates_vs_metrics_line_chart.png")

# Show the line chart
plt.show()

'''
#---------------- Plot the spillage counts for each weight ----------------------
plt.figure(figsize=(12, 6))

spillage_amounts = np.arange(1, 9)  # X-axis values (spillage amounts from 1 to 8)

for i, weight in enumerate(weights):
    plt.plot(spillage_amounts, spillage_counts[i], marker='o', linestyle='-', label=f'Weight = {weight}')

plt.title('Spillage Counts vs Spillage Amounts for Different Weights', fontsize=16)
plt.xlabel('Spillage Amount', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the line chart as an image (optional)
plt.savefig("spillage_counts_line_chart.png")

# Show the line chart
plt.show()

'''