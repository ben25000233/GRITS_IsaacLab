# 🛠️ IsaacLab Spoon Scene Setup

This guide provides step-by-step instructions for building the environment, setting parameters, and running the `interactivate_scene.py` script with a **custom spoon and bowl** configuration in IsaacLab.

---

## 📦 Environment Setup

1. **Install IsaacLab**  
   Make sure the **IsaacLab** package is installed and properly configured.

2. **CUDA Version**  
   - Required: **CUDA 12.1** for IsaacLab

3. **PointNet++ Build Fix (for CUDA 12.1)**  
   When installing **PointNet++** under CUDA 12.1, update the architecture list in  
   `Pointnet2_PyTorch/pointnet2_ops_lib/setup.py`:

   ```python
   # Original
   os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9" 

   # Updated (remove 3.7+PTX)
   os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.6;8.9"
   ```

---

## 🐳 Build and Run Docker

1. **Build Docker Image** (inside `isaaclab_grits` folder)
   ```bash
   ./docker/container.py start
   ```

2. **Run Docker Container**
   ```bash
   docker run --name isaac-lab --entrypoint bash -it --gpus all        -e "ACCEPT_EULA=Y"        -e "PRIVACY_CONSENT=Y"        -e DISPLAY=$DISPLAY        --rm --network=host        -v /tmp/.X11-unix:/tmp/.X11-unix:rw        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw        -v ~/docker/isaac-sim/documents:/root/Documents:rw        -v <isaaclab_grits_path>:/workspace/grits        -v <ckpt_path>:/workspace/ckpt        isaac-lab-base
   ```

---

## ⚙️ Parameter Configuration

> *(Add parameter details here once finalized — e.g., spoon size, bowl position, simulation speed, camera setup.)*

---

## ▶️ Running the Simulation

Inside the container, run:
```bash
python grits_main.py
```
