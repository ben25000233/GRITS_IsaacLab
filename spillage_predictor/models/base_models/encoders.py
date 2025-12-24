import torch.nn as nn
# from models.models_utils import init_weights
import numpy as np
import torch
from spillage_predictor.Pointnet2_PyTorch.pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
# from models.base_models.layers import CausalConv1D, Flatten, conv2d
import torchvision.models as models
import torch.nn.functional as F


class PointNet_Encoder(nn.Module):
    def __init__(self, device,initailize_weights=True):
        super().__init__()
     
        hparams = {
            "model.use_xyz": True,
            "feature_num" : 1
        }
  
        self.model = PointNet2SemSegSSG(hparams).to(device)


    def encode(self, pcd):
        

        if pcd.ndim == 4:
            pcd = pcd.squeeze(0)


        output_pcd = self.model(pcd)

   
        batch_size, x, y = output_pcd.shape
        output_pcd = output_pcd.reshape(batch_size, x* y)

        return output_pcd

    
class flow_pcd_Encoder(nn.Module):
    def __init__(self, device, initailize_weights=True):
        super().__init__()
        
        hparams = {
            "model.use_xyz": True,
            "feature_num" : 4
        }
  
        self.model = PointNet2SemSegSSG(hparams).to(device)
        # if initailize_weights:
        #     init_weights(self.model.modules())


    def encode(self, pcd):
 
        output_pcd = self.model(pcd)
        batch_size, x, y = output_pcd.shape
        output_pcd = output_pcd.reshape(batch_size, x* y)
        
        return output_pcd

    

class ee_pose_Encoder(nn.Module):
    def __init__(self, proprio_dim=9):
        super(ee_pose_Encoder, self).__init__()
        
        self.eepose_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(16, 32),
        )

    def forward(self, eepose):
        
        # Ensure input dtype matches model parameters dtype
        eepose = eepose.to(next(self.eepose_encoder.parameters()).dtype)
        out_eepose = self.eepose_encoder(eepose)
        if out_eepose.ndim == 2 :
            out_eepose = out_eepose.unsqueeze(0)
  
        a,b,c = out_eepose.shape
        out_eepose = out_eepose.reshape(a, b*c)
  
        return out_eepose
    
    
class property_Encoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
      
        super().__init__()

        self.z_dim = z_dim

        self.property_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # if initailize_weights:
        #     init_weights(self.modules())

    def forward(self, property):

        # Ensure input dtype matches model parameters dtype
        property = property.to(next(self.property_encoder.parameters()).dtype)
        
        return self.property_encoder(property).unsqueeze(2)
        
class ImageEncoder(nn.Module):
    def __init__(self, z_dim, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)


        return img_out, img_out_convs

class depth_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.hand_depth_encoder = models.resnet18(pretrained=False)
        self.hand_depth_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.hand_depth_encoder.fc = nn.Linear(self.hand_depth_encoder.fc.in_features, num_classes)
        self.hand_depth_encoder.fc = nn.Identity()
   

    def forward(self, depth):
        # depth encoding layers
        depth_out = self.hand_depth_encoder(depth)
        return depth_out
    

# DP3 encoder
class DP3_Encoder(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=6,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def encode(self, pcd):

        if pcd.dim() != 6 :
            # ----------one-hot label------------------

            if pcd.dim() == 4:        # (B,S,N,4) -> flatten over S for PN++ pass
                B, S, N, C = pcd.shape
                pcd_flat = pcd.reshape(B * S, N, C)
            elif pcd.dim() == 3:      # (B,N,4)
                B, N, C = pcd.shape
                S = 1
                pcd_flat = pcd
            else:
                raise ValueError(f"Unexpected pcd shape {pcd.shape}; expected (B,N,4) or (B,S,N,4)")

            raw_seg = pcd_flat[..., 3]                                 # (B*S, N)
            raw_seg = raw_seg.nan_to_num(0).floor().to(torch.long)

            # build LUT
            lbls = [1, 2, 4]                             # [1,2,4]
            lut = {int(v): i for i, v in enumerate(lbls)}              # 1->0, 2->1, 4->2
            Cseg = len(lut)
            fallback_idx = lut[int(4)] if 4 in lut else 0

            # remap (handles unknowns)
            remapped = torch.full_like(raw_seg, fill_value=fallback_idx)
            for raw, idx in lut.items():
                remapped[raw_seg == raw] = idx

            onehot = F.one_hot(remapped, num_classes=Cseg).to(torch.float32)  # (B*S, N, 3)

            # ---- construct PN++ input: xyz + one-hot ----
            pc_in = torch.cat([pcd_flat[..., :3].to(torch.float32), onehot], dim=-1)  # (B*S, N, 3+3)
        else:
            pc_in = pcd.to(torch.float32)

        x = self.mlp(pc_in)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)

        return x

