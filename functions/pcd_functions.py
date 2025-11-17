import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as Rot
from pytorch3d.transforms import rotation_6d_to_matrix
import torch.nn.functional as F

class Pcd_functions():
    def __init__(self):
        pass
    def depth_to_point_cloud(self, depth, segmantation, object_id = None, object_type = None, device="cpu"):
        
        mask = (segmantation == object_id)
    
        intrinsic = self.compute_camera_intrinsics_matrix(1280, 960, 65)
        h, w = depth.shape
    
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) 
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        if mask is not None:
            points = points[mask.reshape(-1), :]

        # points = self.align_point_cloud(points)
        if object_type == "robot" :
            object_seg = np.full((points.shape[0], 1), 1)
        elif object_type == "food" :
            object_seg = np.full((points.shape[0], 1), 2)
        elif object_type == "bowl":
            object_seg = np.full((points.shape[0], 1), 4)

        # Concatenate the column to the original array
        seg_pcd = np.hstack((points, object_seg))

        return seg_pcd
    def compute_camera_intrinsics_matrix(self,image_width, image_heigth, horizontal_fov):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180
        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)
        K = np.array([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]])
        return K

    def transform_to_world(self,points, extrinsic):

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world_homogeneous = (extrinsic @ points_homogeneous.T).T
        points_world = points_world_homogeneous[:, :3]

        return points_world

    def nor_pcd(self, points):

        seg_info = points[:, 3].reshape(len(points), 1)
        points = points[:, :3]

        # normalize the pcd
        
        centroid = np.mean(points, axis=0)
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))


        centroid = [ 0.59115381, -0.1113387 ,  0.0755547 ]
        m = 0.6797932094342392
        
        points = points - centroid
        points = points / m

        seg_pcd = np.concatenate((points, seg_info), axis=1)
        return seg_pcd

    # def nor_pcd(self, points):
    #     # Split the segmentation info and point coordinates
    #     # seg_info = points[:, 3].view(-1, 1)  # Reshape to (len(points), 1)
    #     seg_info = points[:, 3:].view(-1, points.shape[1] - 3)  # Reshape to (len(points), 3)
    #     points_coords = points[:, :3]

    #     # Define pre-computed centroid and scale factor (m)
    #     centroid = torch.tensor([0.59115381, -0.1113387, 0.0755547], dtype=torch.float32, device=points.device)
    #     m = torch.tensor(0.6797932094342392, dtype=torch.float32, device=points.device)

    #     # Normalize the point cloud
    #     points_coords = points_coords - centroid  # Center the points
    #     points_coords = points_coords / m        # Scale the points

    #     # Concatenate normalized points with segmentation info
    #     seg_pcd = torch.cat((points_coords, seg_info), dim=1)
    #     return seg_pcd
    
    def align_point_cloud(self, points, target_points=10000):
        num_points = len(points)

        np.random.seed(42)
    
        if num_points >= target_points:
            # Randomly downsample to target_points
            indices = np.random.choice(num_points, target_points, replace=False)
            indices = np.sort(indices)

        else:
            # Resample with replacement to reach target_points
            indices = np.random.choice(num_points, target_points, replace=True)
            indices = np.sort(indices)

        new_pcd = np.asarray(points)[indices]
        
        return new_pcd

    
    def check_pcd_color(self, pcd):

        color_map = {
            0: [1, 0, 0],    # Red
            4: [0, 1, 0],    # Green
            1: [0, 0, 1],    # Blue
            3: [1, 1, 0],    # Yellow
            5: [1, 0, 1],     # Magenta
            2: [1, 0.5, 0]
        }
        points = []
        colors = []
    
        
        for i in range(pcd.shape[0]):
            points.append(pcd[i][:3])
            if pcd.shape[1] == 4:
                colors.append(color_map[pcd[i][3]])
            else :
                colors.append([0.5, 0.5, 0.5])


        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])

    def get_translation_matrix(self, x, y, z):
        return torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
        ])

    def from_ee_to_spoon(self, offsets, ee_points):
        """
        offsets: [N, 3]
        ee_points: [B, 9] or [B, 7]  (batch of ee poses)
        returns: [B, N, 3] (transformed point clouds)
        """
        offsets = torch.tensor(offsets, dtype=torch.float32)

        if ee_points.ndim == 1:
            ee_points = torch.tensor(ee_points).unsqueeze(0)
            mode = 'validation'
        else :
            mode = 'train'

        N = offsets.shape[0]
        B = ee_points.shape[0]

        shape_dim = ee_points.shape[1]

        # Precompute constant adjustment matrix
        adjust_matrix = torch.tensor(
            [[1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]],  # Equivalent to Rot.from_euler("XYZ", (0, 180, 180))
            dtype=torch.float32
        )

        # Handle rotation
        if shape_dim == 9:
            rotation_mats = rotation_6d_to_matrix(ee_points[:, 3:9])
        elif shape_dim == 7:
            quat = ee_points[:, 3:7]
            quat = F.normalize(quat, dim=-1)
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            rotation_mats = torch.stack([
                1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw),
                2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw),
                2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)
            ], dim=1).reshape(-1, 3, 3)

        # Apply adjustment matrix
        rotation_mats = rotation_mats.to(dtype=torch.float32)
        rotation_mats = rotation_mats @ adjust_matrix

        # Compute transformed points
        translations = ee_points[:, 0:3]  # [B, 3]

        # Transform all offsets in one go
        rotated_offsets = torch.matmul(rotation_mats.unsqueeze(1), offsets.unsqueeze(-1)).squeeze(-1)  # [B, N, 3]
        new_pcds = rotated_offsets + translations.unsqueeze(1)  # [B, N, 3]

        if mode == 'validation':
            new_pcds = new_pcds.squeeze(0)

        return new_pcds
    
    def get_init_spoon_offset(self, init_pose, init_pcd):
        init_pose = np.array(init_pose)
        # Convert inputs to torch tensors
        offset_list = []
        for i in range(len(init_pcd)):
            offset = init_pcd[i][:3] - init_pose[:3]
            offset_list.append(offset)
        offset_list = np.array(offset_list)
        np.save("real_spoon_pcd_offset.npy", offset_list)

    def show_arrow(self, flow_pcd_list):
        """
        Visualize all flow point clouds at once with arrows indicating flow vectors.
        :param flow_pcd_list: List of Nx6 arrays, where each array contains:
                            - First 3 dimensions: point positions
                            - Last 3 dimensions: flow vectors.
        """

        # Initialize combined points and arrows
        combined_points = []
        combined_lines = []
        combined_colors = []
        arrow_points = []
        line_index = 0

        for flow_pcd in flow_pcd_list:
            # Ensure flow_pcd has the correct shape
            assert flow_pcd.shape[1] == 6, "Each flow_pcd must have 6 dimensions (3 for position, 3 for flow vector)."

            # Extract positions and flow vectors
            positions = flow_pcd[:, :3]
            flow_vectors = flow_pcd[:, 3:]

            # Add points and arrows for the current flow_pcd
            for i in range(len(positions)):
                start_point = positions[i]
                end_point = positions[i] + flow_vectors[i]  # Add flow vector to position
                arrow_points.append(start_point)
                arrow_points.append(end_point)
                combined_lines.append([line_index, line_index + 1])  # Line between start and end
                combined_colors.append([0, 1, 0])  # Green for arrows
                line_index += 2

            # Add the positions to the combined points
            combined_points.extend(positions)

        # Create Open3D point cloud for all positions
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(np.array(combined_points))
        o3d_pcd.paint_uniform_color([1, 0, 0])  # Red for points

        # Create LineSet for all arrows
        arrow_points = np.array(arrow_points)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(arrow_points)
        line_set.lines = o3d.utility.Vector2iVector(np.array(combined_lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(combined_colors))

        # Visualize all point clouds and arrows in one window
        o3d.visualization.draw_geometries([o3d_pcd, line_set])
    

    def eepose_to_flowPcd(self, pcd_offset, eepose_list):
        if eepose_list.ndim == 2:
            eepose_list = eepose_list.unsqueeze(0)
        
        B, T, _ = eepose_list.shape

        # Compute all transformed spoon points
        spoon_pcds = self.from_ee_to_spoon(torch.tensor(pcd_offset, dtype=torch.float32, device=eepose_list.device),
                                                eepose_list.reshape(-1, eepose_list.shape[-1]))
        spoon_pcds = spoon_pcds.reshape(B, T, *spoon_pcds.shape[1:])  # [B, T, N, 3]

        # Compute flow between consecutive steps
        start_pcd = spoon_pcds[:, :-1]  # [B, T-1, N, 3]
        end_pcd = spoon_pcds[:, 1:]     # [B, T-1, N, 3]
        flow = end_pcd - start_pcd
        flow_pcds = torch.cat((start_pcd, flow), dim=-1)  # [B, T-1, N, 6]

        return flow_pcds.float()


        