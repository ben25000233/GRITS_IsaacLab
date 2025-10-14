import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from .base_models.encoders import (
    ee_pose_Encoder,
    property_Encoder,
    PointNet_Encoder,
    flow_pcd_Encoder,
    depth_Encoder,
    DP3_Encoder,
)

from functions.pcd_functions import Pcd_functions

class SensorFusion(nn.Module):
    """
    Regular SensorFusionNetwork Architecture
    Number of parameters:
    Inputs:
        pcd_info:      batch_size x 3 (file_num, pcd_env_num, pcd_index)
        top_pcd:       batch_size x 2000 x 3
        hand_pcd:      batch_size x 2000 x 3
        pcd_index:     batch_size x 1 
        pose_in:       batch_size x 1 x 7
        property_in:   batch_size x 1 x 4
        next_pose_in:  batch_size x 1 x 7
    """

    def __init__(
        self, device , z_dim=128, action_dim=4, encoder=False, deterministic=False, training_type = "spillage"
    ):
        super().__init__()

        self.z_dim = z_dim
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic
        # self.feature_num = 3

        # Modality Encoders
        # self.obs_pcd_encoder = PointNetEncoder(device=device)
        self.obs_pcd_encoder = DP3_Encoder(device=device)
        self.flow_pcd_Encoder = DP3_Encoder(device=device)

        # self.eepose_encoder = ee_pose_Encoder()

        self.pcd_offset = np.load("./ref_pcd/small_tool_offset.npy")
        self.pcd_functions = Pcd_functions()

        # self.pcd_info = np.load("pcd_nor_info.npy", allow_pickle=True)

    def forward_encoder(self, ee_pose, front_pcd):
       

        if front_pcd.ndim == 3 :
            image_num, _, _ = front_pcd.shape
            batch_size = 1
        else : 
            batch_size, image_num, _, _ = front_pcd.shape
  
        obs_pcd = []
        flow_pcd = []
   
        # pose_out = self.eepose_encoder(ee_pose)             # shpae : torch.Size([batch_size , 7, 128])
        

        for i in range(image_num):
            if front_pcd.ndim == 3:
                front_pcd = front_pcd.unsqueeze(0)
            front_pcd_out = self.obs_pcd_encoder.encode(front_pcd[:, i, :, :].unsqueeze(0).float())   # shpae : torch.Size([batch_size , 256])
            obs_pcd.append(front_pcd_out)
        all_obs_pcd = torch.cat(obs_pcd, dim=1)
        a, b = all_obs_pcd.shape
        all_obs_pcd = all_obs_pcd.reshape(batch_size, int(a*b/batch_size))


        flow_pcds = self.pcd_functions.eepose_to_flowPcd(self.pcd_offset, ee_pose)
        for i in range(flow_pcds.shape[1]):
            flow_pcd_out = self.flow_pcd_Encoder.encode(flow_pcds[:, i, :, :].unsqueeze(0).float().to(self.device))   # shpae : torch.Size([batch_size , 256])
            flow_pcd.append(flow_pcd_out)
        all_flow_pcd = torch.cat(flow_pcd, dim=1)
        a, b = all_flow_pcd.shape
        all_flow_pcd = all_flow_pcd.reshape(batch_size, int(a*b/batch_size))
        
        embeddings = torch.cat((all_obs_pcd, all_flow_pcd), 1).to(torch.float32)

        # embeddings = torch.cat((all_obs_pcd, pose_out), 1).to(torch.float32)



        return embeddings
    



class Dynamics_model(SensorFusion):
    """
    SensorFusion Network Architecture without LSTM
    """

    def __init__(
        self, device, z_dim=128, action_dim=9, encoder=False, deterministic=False, training_type="spillage"
    ):
        super().__init__(device, z_dim, action_dim, encoder, deterministic, training_type)
        self.multi_encoder = SensorFusion(device=device)

        # Fully connected layers with BatchNorm
        self.fc1 = nn.Linear(14336, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(256, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(32, 2)  # Output layer for predictions

        # self.projection = nn.Linear(3456, 256)

    def forward(self, ee_pose, tool_with_ball_pcd):
        # Get latent representation from multi-encoder
        latent_z = self.multi_encoder.forward_encoder(ee_pose, tool_with_ball_pcd)

        # Project latent_z to match the dimensions of x
        # projected_latent_z = self.projection(latent_z)

        # Fully connected layers with BatchNorm and residual connection
        x = self.fc1(latent_z)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        # x = x + projected_latent_z  # Add residual connection

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

