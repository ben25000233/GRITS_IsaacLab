import h5py
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d,  matrix_to_quaternion, quaternion_to_matrix



class MultimodalManipulationDataset(Dataset):
    """Multimodal Manipulation dataset with lazy loading."""

    def __init__(
        self,
        filename_list,
        data_length=50,
        training_type="selfsupervised",
        action_dim=7,
        single_env_steps=None, 
        dataset_type="train",
    ):
        """
        Args:
            filename_list (list): List of paths to HDF5 files.
            data_length (int): Length of the data sequence minus one.
            training_type (str): Type of training (e.g., selfsupervised).
            action_dim (int): Dimension of the action space.
            single_env_steps (int): Not used in this implementation.
            type (str): Type of dataset (e.g., 'train' or 'test').
        """

        
        self.single_env_steps = single_env_steps
        self.dataset_path = filename_list
        self.data_length_in_eachfile = data_length 
        self.training_type = training_type
        self.action_dim = action_dim
        self.dataset_type = dataset_type
     
        # We no longer load all the data at once
        self.file_handles = [h5py.File(file, 'r') for file in filename_list]

        

    def __len__(self):
        return len(self.dataset_path) * self.data_length_in_eachfile

    def __getitem__(self, idx):

        # Determine which file and which data entry within that file to load
        file_idx = idx // self.data_length_in_eachfile
        data_index = idx % self.data_length_in_eachfile

        # Open the corresponding file and load the specific data entry
        
        dataset = self.file_handles[file_idx]
        data = self._read_data_from_file(dataset, data_index)

        return data

    def align_point_cloud(self, pcds, target_points=10000):
        new_pcds = []
        for pcd in pcds :
            num_points = pcd.shape[0]
        
            if num_points >= target_points:
                # Randomly downsample to target_points
                indices = np.random.choice(num_points, target_points, replace=False)
                indices = np.sort(indices)

            else:
                # Resample with replacement to reach target_points
                indices = np.random.choice(num_points, target_points, replace=True)
                indices = np.sort(indices)

            new_pcd = np.asarray(pcd)[indices]
            new_pcds.append(new_pcd)
        
        return np.array(new_pcds)

    def _read_data_from_file(self, dataset, idx):
 
        # Read data from a single file for the given index

        # single_num : 8
        # single_num = len(dataset["tool_ball_bowl_pcd"]) / (self.data_length_in_eachfile + 1)
        # print(single_num)
        # exit()
        # current_index = int(idx * single_num) 


        # total predict prame : look back(2) add current frame
        look_back_frame = 3
        future_eepose_num = 12
     
        if idx == 0:
            tool_ball_bowl_pcd = np.tile(dataset["mix_all_pcd"][0] , (look_back_frame, 1, 1)).astype(np.float32)
            # tool_ball_bowl_pcd = self.align_point_cloud(tool_ball_bowl_pcd, target_points=3000)
            if self.dataset_type == "train":
                tool_ball_bowl_pcd = self.jitter_point_cloud(tool_ball_bowl_pcd)
        
        else : 
        
            begin_idx = -look_back_frame 
            tool_ball_bowl_pcd = dataset["mix_all_pcd"][(idx-1)*future_eepose_num : idx*future_eepose_num][begin_idx:].astype(np.float32)
            # tool_ball_bowl_pcd = self.align_point_cloud(tool_ball_bowl_pcd, target_points=3000)
            if self.dataset_type == "train":
                tool_ball_bowl_pcd = self.jitter_point_cloud(tool_ball_bowl_pcd)

        # future ee_pose and tool_pcd
        
        eepose = dataset["real_eepose"][idx*future_eepose_num: (idx+1)*future_eepose_num]
        
        binary_label = dataset["binary_spillage"][idx]
    
        rotation_6d_pose = qua_to_rotation_6d(eepose)
        nor_6d_pose = ee_normalize(rotation_6d_pose)

        if dataset["shape"][()] == b'sphere':
            shape = "sphere"
        elif dataset["shape"][()] == b'cube':
            shape = "cube"
        elif dataset["shape"][()] == b'cylinder':     
            shape = "cylinder"
        elif dataset["shape"][()] == b'cone':     
            shape = "cone"
        
      
        single_data = {
            "eepose": nor_6d_pose,
            # "ee_pcd" : tool_pcd,
            # "spillage_type": spillage_index,
            # "tool_with_ball_pcd" : tool_with_ball_pcd, 
            # "pcd_with_flow" : pcd_with_flow,
            "binary_label" : binary_label,
            "tool_ball_bowl_pcd" : tool_ball_bowl_pcd,
            "shape": shape,
        }

        return single_data
    

    def cal_transformation(self, p1, p2, pcd, eepoint1):

        pos1 = p1.copy()
        pos2 = p2.copy()

        bias = [-0.15,-0.15,0.03]
        temp = pos1[:3] - (eepoint1[:3] - bias) 
        pos1[:3] -= temp
        pos2[:3] -= temp
        
        pcd_point = pcd[:, :3]
        
        pcd_seg = pcd[:,3]
        from scipy.spatial.transform import Rotation as R
        r1 = pos1[:4]
        r2 = pos2[:4]
        # r1 = np.array([pose1[3], pose1[0],pose1[1],pose1[2]])
        # r2 = np.array([pose2[3], pose2[0],pose2[1],pose2[2]])
        rot1 = R.from_quat(r1).as_matrix()  # Rotation matrix from quat1
        rot2 = R.from_quat(r2).as_matrix()  # Rotation matrix from quat2
        # Compute the relative transformation
        relative_rotation = rot2 @ rot1.T      # Relative rotation matrix
   
        # relative_rotation[0, 2] = -relative_rotation[0, 2]  # Negate sin(theta)
        # relative_rotation[2, 0] = -relative_rotation[2, 0]  # Negate -sin(theta)

        base_point = pos1[:3]
        pcd_point = pcd_point - base_point
       
        relative_translation = pos2[:3] - (relative_rotation @ pos1[:3])  # Relative translation

        transformed_point = (relative_rotation @ pcd_point.T).T  + relative_translation + base_point
  
        trans_pcd = np.concatenate((transformed_point, pcd_seg.reshape(pcd_seg.shape[0], 1)), axis = 1)

        return trans_pcd

    def jitter_point_cloud(self, pcds, sigma=0.002, clip=0.05):
        """
        pcd: (N, 3) or (B, N, 3) numpy array
        """
        noise_pcd = []
        for pcd in pcds:
            
            noise = np.clip(sigma * np.random.randn(*pcd.shape), -clip, clip)
            noise[..., 3:] = 0
            noise_pcd.append(pcd + noise)

        return np.array(noise_pcd)
        

    def __del__(self):
        # Close all file handles when the dataset object is deleted
        for file_handle in self.file_handles:
            file_handle.close()


def get_translation_matrix(x, y, z):
    return torch.tensor([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]
    ])

def qua_to_rotation_6d(ee_traj):

    ee_traj = torch.tensor(ee_traj, dtype=torch.float32)

    rotation_6d_traj = torch.zeros((ee_traj.shape[0], 9))  # (H, 9)
    
    for i in range(ee_traj.shape[0]):
        ee = ee_traj[i]
        quaternion = ee[3:]  # (qw, qx, qy, qz)

        # Convert quaternion to rotation matrix
        ee_rotation_matrix = quaternion_to_matrix(quaternion)

        # adjust_matrix = R.from_quat(quaternion).as_matrix()

        # print(ee_rotation_matrix)
        # print(adjust_matrix)
        # exit()


        rotation_6d_traj[i, :3] = ee[:3]
        rotation_6d_traj[i, 3:] = matrix_to_rotation_6d(ee_rotation_matrix[0:3, 0:3])

    return np.array(rotation_6d_traj)  # (H, 9)

def ee_normalize(data):
    data = torch.tensor(data, dtype=torch.float32)
    
    # Correctly load the input range tensor
    input_range = torch.load('input_range_sim.pt', weights_only = True)  # Removed invalid argument

    input_max = input_range[0, :]
    input_min = input_range[1, :]
    # self.input_mean = input_range[2, :]  # Uncomment if mean is needed elsewhere
    ranges = input_max - input_min

    data_normalize = torch.zeros_like(data)

    for i in range(3):  # Normalize only the first three columns
        if ranges[i] < 1e-4:
            # If variance is small, shift to zero-mean without scaling
            data_normalize[:, i] = data[:, i] - input_min[i]
        else:
            # Scale to [-1, 1] range
            data_normalize[:, i] = -1 + 2 * (data[:, i] - input_min[i]) / ranges[i]

    # Preserve the remaining columns as-is
    data_normalize[:, 3:] = data[:, 3:]

    return np.array(data_normalize)
