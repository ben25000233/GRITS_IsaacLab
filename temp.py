import random
import numpy as np
import yaml
from pyconfigparser import configparser


# Generate 50 unique noise pairs
noise_pairs = []
while len(noise_pairs) < 50:
    noise_1 = round(random.uniform(0, 0.02), 4)
    noise_2 = round(random.uniform(0, 0.02), 4)
    pair = (noise_1, noise_2)
    if pair not in noise_pairs:  # Check for duplicates
        noise_pairs.append(pair)  # Add the pair to the list

# Convert the list to a NumPy array for saving
noise_pairs_array = np.array(noise_pairs)

# Convert the NumPy array to a list for YAML compatibility
noise_pairs_list = noise_pairs_array.tolist()

print(len(noise_pairs_list))  # Print the number of unique pairs

# Save the list to a YAML file
with open("noise_pairs.yaml", "w") as yaml_file:
    yaml.dump({"noises": noise_pairs_list}, yaml_file, default_flow_style=False)

print("Noise pairs saved to noise_pairs.yaml")

cfg = configparser.get_config(config_dir = "", file_name="noise_pairs.yaml") 
print(np.array(cfg.noises).shape)
