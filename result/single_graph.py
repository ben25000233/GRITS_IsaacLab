import numpy as np
import json
import matplotlib.pyplot as plt



file_name = f"./long_denoise/R0.009_M0.003_A9_F0.5_Ssphere_W3.json"
with open(file_name, "r") as json_file:
    spillage_data = json.load(json_file)


all_array = np.array(spillage_data["spillage_scoop"])
spillage_array = all_array[:,0]


spillage_amount = np.count_nonzero(spillage_array)
scoop_success_array = all_array[all_array[:, 1] > 0]
success_array = scoop_success_array[scoop_success_array[:, 0] == 0]
print(f"scoop_failure_rate : {(len(all_array) - len(scoop_success_array))/len(all_array)}")
print(f"spillage_rate : {spillage_amount/len(all_array)}")
# print(f"Mean of spillage : {np.mean(spillage_array)}")
print(f"success_rate : {len(success_array)/len(all_array)}")



chart_type = "bar"

if chart_type == "bar":
    bins = np.arange(0, 10) + 0.5 
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    counts, bin_edges, bars = plt.hist(
        spillage_array, bins=bins, color="b", alpha=0.7, edgecolor="black", label="Spillage Amounts"
    )

    # Annotate the histogram with spillage rates
    for count, bar in zip(counts, bars):
        height = bar.get_height()
        if height > 0:  # Only annotate non-zero bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",  # Display the count as an integer
                ha="center",
                va="bottom",
                fontsize=18,
                color="black",
            )

    plt.xlim(0, 10)
    plt.ylim(0, 30)
    plt.title("Spillage Amount Distribution", fontsize=16)
    plt.xlabel("Spillage Amount", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=6)
    plt.tight_layout()

    # Save the histogram as an image (optional)
    plt.savefig(f"{file_name}_spillage_histogram.png")

    # Show the histogram
    # plt.show()
elif chart_type == "pie":
    # Count occurrences of each unique value
    unique_values, counts = np.unique(spillage_array, return_counts=True)

    # Create labels using unique values
    labels = [f"{value}" for value in unique_values]

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1i', startangle=90, colors=plt.cm.tab20.colors)
    plt.title('Spillage Amount Distribution', fontsize=16)

    # Save the pie chart as an image (optional)
    plt.savefig(f"{file_name}_spillage_pie_chart.png")

    # Show the pie chart
    plt.show()
elif chart_type == "line":

    # Count occurrences of each unique value
    unique_values, counts = np.unique(spillage_array, return_counts=True)

    # Plot the line chart
    plt.figure(figsize=(10, 6))
    plt.plot(unique_values, counts, marker='o', linestyle='-', color='b', label='Spillage Amounts')
    plt.title('Spillage Amount Distribution', fontsize=16)
    plt.xlabel('Spillage Amount', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the line chart as an image (optional)
    plt.savefig(f"{file_name}_spillage_line_chart.png")

    # Show the line chart
    plt.show()