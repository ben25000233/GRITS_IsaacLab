import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as Rot

class functions():
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.eepose_offset = 0.035
        # new_offset = self.pcd_offset
        # new_offset[:, 2] -= (0.1- self.eepose_offset)
        # new_offset[:, 1] += 0.03
        # np.save("./ref_pcd/temp_offset.npy", new_offset)
        # simulation_app.close()

    def eepose_sim2real_offset(self, sim_qua_list):

        update_qua_list = []

        for sim_qua in sim_qua_list:
            real_quat_rotation_xyzw = np.array([sim_qua[4], sim_qua[5], sim_qua[6], sim_qua[3]])
            rotation = Rot.from_quat(real_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()
            y_rot = rotation_matrix[:, 1]
            z_rot = rotation_matrix[:, 2]
            updata_qua_pose = [sim_qua[0], sim_qua[1], sim_qua[2]] + z_rot * self.eepose_offset + y_rot * 0.03

            update_qua = np.array([updata_qua_pose[0], updata_qua_pose[1], updata_qua_pose[2], sim_qua[3], sim_qua[4], sim_qua[5], sim_qua[6]])
            update_qua_list.append(update_qua)

        return torch.tensor(np.array(update_qua_list)).to(self.device)


    def eepose_real2sim_offset(self, real_qua_list):
        if len(real_qua_list) != 1:
            real_qua_list = real_qua_list.to("cpu")
        update_qua_list = []

        for real_qua in real_qua_list:
            real_quat_rotation_xyzw = np.array([real_qua[4], real_qua[5], real_qua[6], real_qua[3]])
            rotation = Rot.from_quat(real_quat_rotation_xyzw)
            rotation_matrix = rotation.as_matrix()

            x_rot = rotation_matrix[:, 0]
            y_rot = rotation_matrix[:, 1]
            z_rot = rotation_matrix[:, 2]
            updata_qua_pose = [real_qua[0], real_qua[1], real_qua[2]] - y_rot * 0.03 - z_rot * self.eepose_offset

            update_qua = np.array([updata_qua_pose[0] , updata_qua_pose[1], updata_qua_pose[2], real_qua[3], real_qua[4], real_qua[5], real_qua[6]])
            update_qua_list.append(update_qua)

        return torch.tensor(np.array(update_qua_list)).to(self.device)
    
    def list_to_nparray(self, lists):
        temp_array = []

        for i in range(len(lists)):
            temp_array.append(np.array(lists[i]))

        temp = np.stack(temp_array, axis=0)

        shape = temp.shape
        new_shape = (shape[0] * shape[1],) + shape[2:]
        temp_1 = temp.reshape(new_shape )
 
        return temp_1
    
    