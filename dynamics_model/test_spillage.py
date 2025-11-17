from __future__ import print_function
import time

import numpy as np
import torch

from .models.sensor_fusion import Dynamics_model

class spillage_predictor:
    def __init__(self):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        if use_cuda:
            print("Let's use", torch.cuda.device_count(), "GPUs!")


        # model
        self.model = Dynamics_model(
            device=self.device,
        ).to(self.device)

        # model_path = "./dynamics_model/spillage_ckpt.pt"
        model_path = "/media/hcis-s22/data/isaaclab_spillage_dataset/fix_tool_bowl_dataset/all_train/sim_experiment/spillage_ckpt/epoch100.pt"
        print("Loading model from {}...".format(model_path))
        ckpt = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(ckpt)


    def validate(self, eepose, tool_with_ball_pcd):
       
        self.model.eval()
        pred_spillage = self.model(eepose, tool_with_ball_pcd)
        
        return pred_spillage
    
    

    

    