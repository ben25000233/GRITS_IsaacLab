# 🛠️ IsaacLab Spoon Scene Setup

This guide explains how to build the environment and set up parameters for running the `interactivate_scene.py` script with a custom spoon and bowl setup in IsaacLab.

---

## 📦 Build Environment

Ensure the **IsaacLab** package is installed and properly set up.

cuda version : 12.1 for isaaclab

for pytorch3d installation :
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .


for point++ installation under cuda 12.1
in Pointnet2_PyTorch/pointnet2_ops_lib/setup.py : 
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9" 
=> os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9" 



---

## ⚙️ Set Parameters



---
## ▶️ Run the Simulation

