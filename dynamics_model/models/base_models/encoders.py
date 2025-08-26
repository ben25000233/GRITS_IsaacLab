import torch.nn as nn
# from models.models_utils import init_weights
import numpy as np
import torch
from dynamics_model.Pointnet2_PyTorch.pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
# from models.base_models.layers import CausalConv1D, Flatten, conv2d
import torchvision.models as models
import torch.nn.functional as F


    
class obs_pcd_Encoder(nn.Module):
    def __init__(self, device,initailize_weights=True):
        super().__init__()
     
        hparams = {
            "model.use_xyz": True,
            "feature_num" : 1
        }
  
        self.model = PointNet2SemSegSSG(hparams).to(device)

        self.backbone = PointNet2SemSegSSG(hparams).to(device)

        self.backbone.fc_layer = nn.Identity()
        # Global pooling + projection to an embedding
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 256)
        )
        # if initailize_weights:
        #     init_weights(self.model.modules())


    def encode(self, pcd):
        

        if pcd.ndim == 4:
            pcd = pcd.squeeze(0)

        output_pcd = self.model(pcd)

   
        batch_size, x, y = output_pcd.shape
        output_pcd = output_pcd.reshape(batch_size, x* y)
        
        return output_pcd

'''
    def encode(self, pcd):
        # pcd expected: (B, N, C). If (B,S,N,C), pick last S or fuse before here.
        if pcd.ndim == 4:
            pcd = pcd.squeeze(0)
  

        # Backbone forward to get (B, 512, npoint)
        xyz, features = self.backbone._break_up_pc(pcd)  # reuse helper

        for m in self.backbone.SA_modules:
            xyz, features = m(xyz, features)


        # Global pool across points → (B, 512)
        feat = self.pool(features)  # (B, 512)
        emb = self.proj(feat.squeeze(-1))

        # ----------one-hot label------------------

        # if pcd.dim() == 4:        # (B,S,N,4) -> flatten over S for PN++ pass
        #     B, S, N, C = pcd.shape
        #     pcd_flat = pcd.reshape(B * S, N, C)
        # elif pcd.dim() == 3:      # (B,N,4)
        #     B, N, C = pcd.shape
        #     S = 1
        #     pcd_flat = pcd
        # else:
        #     raise ValueError(f"Unexpected pcd shape {pcd.shape}; expected (B,N,4) or (B,S,N,4)")

        # raw_seg = pcd_flat[..., 3]                                 # (B*S, N)
        # raw_seg = raw_seg.nan_to_num(0).floor().to(torch.long)

        # # build LUT
        # lbls = [1, 2, 4]                             # [1,2,4]
        # lut = {int(v): i for i, v in enumerate(lbls)}              # 1->0, 2->1, 4->2
        # Cseg = len(lut)
        # fallback_idx = lut[int(4)] if 4 in lut else 0

        # # remap (handles unknowns)
        # remapped = torch.full_like(raw_seg, fill_value=fallback_idx)
        # for raw, idx in lut.items():
        #     remapped[raw_seg == raw] = idx

        # onehot = F.one_hot(remapped, num_classes=Cseg).to(torch.float32)  # (B*S, N, 3)

        # # ---- construct PN++ input: xyz + one-hot ----
        # pc_in = torch.cat([pcd_flat[..., :3].to(torch.float32), onehot], dim=-1)  # (B*S, N, 3+3)

        # # ---- backbone forward (use the correct tensor!) ----
        # xyz, features = self.backbone._break_up_pc(pc_in)  # xyz: (B*S,N,3), features: (B*S,3,N)
        # for m in self.backbone.SA_modules:
        #     xyz, features = m(xyz, features)              # features: (B*S, 512, npoint)

        # # ---- global pool & project ----
        # feat = self.pool(features).squeeze(-1)            # (B*S, 512)
        # emb  = self.proj(feat)
    
        return emb

'''
    
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
    

    

