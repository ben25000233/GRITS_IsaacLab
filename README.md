# 🛠️ IsaacLab Spoon Scene Setup

This repository provides step-by-step instructions for setting up an IsaacLab environment and running simulations with a **custom spoon and bowl** configuration. It covers environment preparation, Docker usage, parameter configuration, and script execution for data collection and GRITS experiments.

---

## 📦 Environment Setup

### 1. Install IsaacLab
Ensure that **IsaacLab** is correctly installed and configured on your system.

### 2. CUDA Version
- **Required:** CUDA **12.1**

### 3. PointNet++ Build Fix (CUDA 12.1)
When installing **PointNet++** with CUDA 12.1, you must update the CUDA architecture list to avoid build errors.

Edit the following file:
```
Pointnet2_PyTorch/pointnet2_ops_lib/setup.py
```

Update the architecture list as shown below:

```python
# Original
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9"

# Updated (remove 3.7+PTX)
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9"
```

---

## 🐳 Build and Run Docker

### 1. Build the Docker Image
From inside the `isaaclab_grits` directory, start the container build process:

```bash
./docker/container.py start
```

---

### 2. Run Docker Container (Headless Mode)
Please update the mounted directory path to match your local folder. 
```bash
docker run \
    --name GRITS \
    --entrypoint bash \
    -it \
    --gpus all \
    --rm \
    --shm-size="24g" \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=" \
    -e "USE_EGL=1" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /home/hcis-s22/benyang/scoop-env/isaaclab_grits:/workspace/grits \
    -v /media/hcis-s22/data/dp_experiments/0901_sim/ckpt:/workspace/dp_ckpt \
    -v /media/hcis-s22/data/isaaclab_spillage_dataset/fix_tool_bowl_dataset/all_train:/workspace/train_spillage_dataset \
    -v /media/hcis-s22/data/isaaclab_spillage_dataset/fix_tool_bowl_dataset/all_validation:/workspace/val_spillage_dataset \
    -v /media/hcis-s22/data/isaaclab_dp_split_dataset:/workspace/dp_dataset \
    isaac-lab-base
```

---

### 3. Run Docker Container (GUI Mode)

Enable X11 access:
```bash
xhost +local:docker
```

Run the container:
```bash
docker run \
    --name GRITS \
    --entrypoint bash \
    -it \
    --gpus all \
    --shm-size="24g" \
    --rm \
    --network=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v /home/hcis-s22/benyang/scoop-env/isaaclab_grits:/workspace/grits \
    -v /media/hcis-s22/data/dp_experiments/0901_sim/ckpt:/workspace/dp_ckpt \
    -v /media/hcis-s22/data/isaaclab_spillage_dataset/fix_tool_bowl_dataset/all_train:/workspace/train_spillage_dataset \
    -v /media/hcis-s22/data/isaaclab_spillage_dataset/fix_tool_bowl_dataset/all_validation:/workspace/val_spillage_dataset \
    -v /media/hcis-s22/data/isaaclab_dp_split_dataset:/workspace/dp_dataset \
    isaac-lab-base
```

---

## ⚙️ Parameter Configuration

Modify experiment parameters in:
```
config/grits.yaml
```
IsaacLab_env setting functions:
```
function/Env_functions.py
```
---

## ▶️ Running the Simulation

### Diffusion Policy (DP) Demonstration Collection
```bash
python isaaclab_dp_data_collect.py
```

### Spillage Dataset Collection
```bash
python isaaclab_spillage_data_collect.py
```

### Train spillge predictor
```bash
python dynamic_training.py
```

### Run GRITS
```bash
python grits_main.py
```
