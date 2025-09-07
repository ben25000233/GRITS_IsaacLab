import random
import numpy as np
import yaml
from pyconfigparser import configparser


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

        return np.array(update_qua_list)


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

        return np.array(update_qua_list)