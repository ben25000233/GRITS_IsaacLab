import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import matplotlib.pyplot as plt


import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
import isaacsim.core.utils.prims as prim_utils

from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.converters import UrdfConverterCfg


from dynamics_model.test_spillage import spillage_predictor
from dp_model.predict import LfD
from pytorch3d.transforms import matrix_to_rotation_6d, quaternion_to_matrix
from pyconfigparser import configparser

from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
import yaml
import json
from functions.pcd_functions import Pcd_functions
from functions.Env_functions import TableTopSceneCfg
from functions.functions import functions



class Grits():
    def __init__(self, init_pose = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.spillage_predictor = spillage_predictor()
        self.lfd = LfD()

        self.franka_init_pose = init_pose
    

        self.gt_front = np.load("./real_cam_pose/front_cam2base.npy")
        self.gt_back = np.load("./real_cam_pose/back_cam2base.npy")

        self.ref_bowl = np.load("./ref_pcd/ref_bowl_pcd.npy")
        self.real_food = np.load("./ref_pcd/real_food.npy")

        self.init_spoon_pcd = np.load("./ref_pcd/ref_spoon_pcd.npy")
        self.pcd_offset = np.load("./ref_pcd/real_spoon_pcd_offset.npy")

        self.eepose_offset = 0.035


        self.pcd_offset = np.load("./ref_pcd/temp_offset.npy")
        
        self.pcd_functions = Pcd_functions()
        self.functions = functions()

        self.num_envs = 1

        self.robot_semantic_id = None
        self.bowl_semantic_id = None
        self.food_semantic_id = None

        self.action_horizon = 12

        # self.r_radius, self.l_radius, self.mass, self.friction  = food_info
        

        # robot proprioception data
        self.record_ee_pose = [[] for _ in range(self.num_envs)]

        # front camera data
        self.front_rgb_list = [[] for _ in range(self.num_envs)]
        self.front_depth_list = [[] for _ in range(self.num_envs)]
        self.front_seg_list = [[] for _ in range(self.num_envs)]
        self.mix_all_pcd_list = [[] for _ in range(self.num_envs)]

        # back camera data
        self.back_rgb_list = [[] for _ in range(self.num_envs)]
        self.back_depth_list = [[] for _ in range(self.num_envs)]
        self.back_seg_list = [[] for _ in range(self.num_envs)]
        self.back_pcd_color_list = [[] for _ in range(self.num_envs)]

        self.spillage_amount = [[] for _ in range(self.num_envs)]
        self.scooped_amount = [[] for _ in range(self.num_envs)]
        self.spillage_vol = [[] for _ in range(self.num_envs)]
        self.scooped_vol = [[] for _ in range(self.num_envs)]
        self.spillage_type = [[] for _ in range(self.num_envs)]
        self.scooped_type = [[] for _ in range(self.num_envs)]
        self.binary_spillage = [[] for _ in range(self.num_envs)]
        self.binary_scoop = [[] for _ in range(self.num_envs)]
        self.pre_spillage = np.zeros(self.num_envs)



    def get_info(self, robot_entity_cfg = None):


        front_rgb_image  = self.front_camera.data.output["rgb"][0].cpu().numpy()
        front_depth_image  = self.front_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        front_seg_image  = self.front_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        back_rgb_image  = self.back_camera.data.output["rgb"][0].cpu().numpy()
        back_depth_image  = self.back_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        back_seg_image  = self.back_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        # plt.imshow(back_rgb_image)
        # plt.show()
        # simulation_app.close()


        food_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "food", object_id = self.food_semantic_id)
        back_food_world = self.pcd_functions.transform_to_world(food_pcd[:, :3], self.gt_back)
        object_seg = np.full((back_food_world.shape[0], 1), 2)
        back_food_world = np.hstack((back_food_world, object_seg))



        bowl_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "bowl", object_id = self.bowl_semantic_id)
        back_bowl_world = self.pcd_functions.transform_to_world(bowl_pcd[:, :3], self.gt_back)
        # object_seg must be 4
        object_seg = np.full((back_bowl_world.shape[0], 1), 3)
        back_bowl_world = np.hstack((back_bowl_world, object_seg))


        # bowl_pcd = self.pcd_functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "bowl", object_id = self.bowl_semantic_id)
        # front_bowl_world = self.pcd_functions.transform_to_world(bowl_pcd[:, :3], self.gt_front)
        # object_seg = np.full((front_bowl_world.shape[0], 1), 4)
        # front_bowl_world = np.hstack((front_bowl_world, object_seg))
        # front_bowl_world = self.pcd_functions.align_point_cloud(front_bowl_world, target_points = 10000)
        # np.save(f"ref_bowl_pcd.npy", front_bowl_world) 
        # self.pcd_functions.check_pcd_color(front_bowl_world)
        # simulation_app.close()


        # get eepose
        sim_current_pose = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # real_current_pose = self.functions.eepose_sim2real_offset(sim_current_pose.to("cpu"))
        self.record_ee_pose[0].append(sim_current_pose[0].to("cpu"))

        
        # get front_tool for get ref spoon pcd 
        tool_pcd = self.pcd_functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "robot", object_id = self.robot_semantic_id)
        front_tool_world = self.pcd_functions.transform_to_world(tool_pcd[:, :3], self.gt_front)
        front_tool_world = front_tool_world[front_tool_world[:, 0] > 0.1]
        # front_tool_world = front_tool_world[front_tool_world[:, 2] < 0.2]
        # front_tool_world = self.pcd_functions.align_point_cloud(front_tool_world, target_points = 10000)

        object_seg = np.full((front_tool_world.shape[0], 1), 5)
        front_tool_world = np.hstack((front_tool_world, object_seg))
        
        # self.pcd_functions.check_pcd_color(front_tool_world)
        # simulation_app.close()
    


        # get back_tool for view
        tool_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "robot", object_id = self.robot_semantic_id)
        back_tool_world = self.pcd_functions.transform_to_world(tool_pcd[:, :3], self.gt_back)
        back_tool_world = back_tool_world[back_tool_world[:, 0] > 0.1]
        object_seg = np.full((back_tool_world.shape[0], 1), 3)
        back_tool_world = np.hstack((back_tool_world, object_seg))
        

        
        # trans_tool = self.pcd_functions.from_ee_to_spoon(self.pcd_offset, real_current_pose[0], self.init_spoon_pcd)

        sim_ee_pose = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        real_eepose = self.functions.eepose_sim2real_offset(sim_ee_pose.to("cpu"))[0]

        trans_tool = self.pcd_functions.from_ee_to_spoon(self.pcd_offset, real_eepose)

        # np.save(f"ref_spoon_pcd.npy", front_tool_world)
        # self.pcd_functions.get_init_spoon_offset(real_eepose.to("cpu"), front_tool_world)
        # simulation_app.close()

        object_seg = np.full((trans_tool.shape[0], 1), 1)
        trans_tool = np.hstack((trans_tool, object_seg))
 

        # mix_all_pcd = np.concatenate(( trans_tool, back_food_world, self.ref_bowl), axis=0)
        mix_all_pcd = np.concatenate((trans_tool, back_tool_world, back_bowl_world, self.real_food), axis=0)
        mix_all_pcd = self.pcd_functions.align_point_cloud(mix_all_pcd, target_points = 30000)
        mix_all_nor_pcd = self.pcd_functions.nor_pcd(mix_all_pcd)
        # self.pcd_functions.check_pcd_color(mix_all_nor_pcd)
        # simulation_app.close()

    
    
        self.front_rgb_list[0].append(front_rgb_image)
        self.front_depth_list[0].append(front_depth_image)
        self.front_seg_list[0].append(front_seg_image)

        self.back_rgb_list[0].append(back_rgb_image)
        self.back_depth_list[0].append(back_depth_image)
        self.back_seg_list[0].append(back_seg_image)

        self.mix_all_pcd_list[0].append(mix_all_nor_pcd)

    def move_generate(self):
  
        if len(self.back_rgb_list[0]) == 1:

            begin_rgb = self.back_rgb_list[0].copy()
      
            images = [np.array(begin_rgb[0])] * 5
            depths = [np.array(self.back_depth_list[0])[0].astype(np.float32)] * 5 
            eepose = [np.array(self.record_ee_pose[0])[0].astype(np.float32)] * 5
            seg_pcd = [np.array(self.mix_all_pcd_list[0])[0]]*5

            
        else:
            images = self.back_rgb_list[0][-5:]  
            depths = self.back_depth_list[0][-5 :]
            eepose = self.record_ee_pose[0][-5:]
            seg_pcd = self.mix_all_pcd_list[0][-5:]

        
        image_array = np.array(images)[:, :800, 80:]
        depth_array = np.array(depths)[:, :800, 80:]
        eepose_array = np.array(eepose)
        seg_pcd_array = np.array(seg_pcd)

        eeposes = self.lfd.run_model(image_array, depth_array, eepose_array, seg_pcd_array)

        return eeposes

    
    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        self.robot = scene["robot"]
        self.front_camera = scene["front_camera"]
        self.back_camera = scene["back_camera"]
        self.device = sim.device

        reset_frame = 100


        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

        # Markers
        # frame_marker_cfg = FRAME_MARKER_CFG.copy()
        # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))



        # Create buffers to store actions
        ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=self.device)
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_finger_left"])
        
        # Resolving the scene entities
        robot_entity_cfg.resolve(scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self.robot.is_fixed_base:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            ee_jacobi_idx = robot_entity_cfg.body_ids[0]

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0

        frame_num = 0
        current_goal_idx = 0
        goal_pose = None
        action = None


        id_to_labels = self.back_camera.data.info[0]["semantic_segmentation"]["idToLabels"]
        # print(id_to_labels)
        for semantic_id_str, label_info in id_to_labels.items():
            if label_info.get("class") == "bowl":
                self.bowl_semantic_id = int(semantic_id_str)
            if label_info.get("class") == "food":
                self.food_semantic_id = int(semantic_id_str)
                # print(f"food semantic id: {self.food_semantic_id}")
            if label_info.get("class") == "robot":
                self.robot_semantic_id = int(semantic_id_str)
            
        # Simulation loop
        while simulation_app.is_running():
            # init set
            # print(f"frame_num: {frame_num}")
            ## joint state
            # print(self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids])
            ## eepose
            # print(self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7])
        
            if frame_num <= reset_frame:
         
                init_joint =  torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]], device = sim.device)

                if frame_num == reset_frame :
                    init_joint =  self.franka_init_pose
                    self.cal_spillage_scooped(scene = scene, reset = 1)

                joint_vel = self.robot.data.default_joint_vel.clone()
                joint_pos = init_joint
            
                joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            
                self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
                self.robot.reset()
            
            else :
                # scooping speed
                if frame_num % 5 == 0:
                    
                    self.get_info(robot_entity_cfg)

                    if current_goal_idx % self.action_horizon == 0 :
                        action = self.move_generate()
                        
                        print("current_goal_idx : ", current_goal_idx)
                        self.cal_spillage_scooped(scene = scene, reset = 0)

                        if current_goal_idx != 0 : 
                            
                            pcd_list = torch.tensor(np.array(self.mix_all_pcd_list[0][-3:]), dtype = torch.float32).to("cuda:0")
                            traj = self.qua_to_rotation_6d(action).to("cuda:0")
                
                            spillage_logic = self.spillage_predictor.validate(traj, pcd_list)
                            spillage_prob = torch.nn.functional.softmax(spillage_logic[0], dim=-1)[1]

                            print(spillage_prob)
                            # self.pcd_functions.check_pcd_color(np.array(pcd_list[0].to("cpu")))
                            # simulation_app.close()
                    
                    goal_pose = torch.tensor(action[current_goal_idx % self.action_horizon]).to(self.device)
                    
                    joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    ik_commands[:] = goal_pose
                    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                    # # reset controller
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(ik_commands)

                    # change goal
                    current_goal_idx += 1

                    if current_goal_idx >= 144 :
                        break
              
                    
            
            # obtain quantities from simulation
            jacobian = self.robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


            frame_num += 1  


            # apply actions
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()

            sim_time += sim_dt
            # update sim-time
            # update buffers
            scene.update(sim_dt)

            '''
            # vis offset
            ee_pose_w = self.robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            ee_pose_w = self.functions.eepose_sim2real_offset(ee_pose_w.to("cpu"))
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ee_goals[current_goal_idx].unsqueeze(0)[:, 0:3], ee_goals[current_goal_idx].unsqueeze(0)[:, 3:7])
            '''

    def cal_spillage_scooped(self, env_index = 0, reset = 0, scene = None):
        # reset = 1 means record init spillage in experiment setting 
        current_spillage = 0
        scoop_amount = 0

        rigid_object = scene["rigid_object"].data.body_link_state_w
        y_pose = rigid_object[:,0, 1].to("cpu")
        z_pose = rigid_object[:,0, 2].to("cpu")

        spillage_mask = np.logical_or(z_pose < 0, y_pose > -0.02)
        current_spillage = np.count_nonzero(spillage_mask)

        scoop_mask = np.logical_or(z_pose > 0.15, np.logical_and(z_pose > 0, y_pose > 0))
        scoop_amount = np.count_nonzero(scoop_mask)

        if reset == 1:
            init_amount_mask = z_pose > 0.03
            self.init_amount_amount = np.count_nonzero(init_amount_mask)
   

        
        if reset == 0:
         
            spillage_amount = current_spillage - self.pre_spillage[env_index]
            # spillage_vol = spillage_amount * (self.ball_radius**3) * 10**9
            # scoop_vol = scoop_amount * (self.ball_radius**3)* 10**9
            
            if int(spillage_amount) == 0:
                self.binary_spillage[env_index].append(0)
            else :
                self.binary_spillage[env_index].append(1)
    
            if int(scoop_amount) == 0:
                self.binary_scoop[env_index].append(0)
            else :
                self.binary_scoop[env_index].append(1)
          

            self.spillage_amount[env_index].append(int(spillage_amount))
            self.scooped_amount[env_index].append(int(scoop_amount))
            # self.spillage_vol[env_index].append(int(spillage_vol))
            # self.scooped_vol[env_index].append(int(scoop_vol))


            print(f"spillage amount :{int(spillage_amount)}")
            print(f"scoop_num : {int(scoop_amount)}")
       
        self.pre_spillage[env_index] = int(current_spillage)
        
    
    def qua_to_rotation_6d(self, ee_traj):
   
        ee_traj = torch.tensor(np.array(ee_traj), dtype=torch.float32)

        rotation_6d_traj = torch.zeros((ee_traj.shape[0], 9))  # (H, 9)
        
        for i in range(ee_traj.shape[0]):
            ee = ee_traj[i]
            quaternion = ee[3:]  # (qw, qx, qy, qz)

            # Convert quaternion to rotation matrix
            ee_rotation_matrix = quaternion_to_matrix(quaternion)

            rotation_6d_traj[i, :3] = ee[:3]
            rotation_6d_traj[i, 3:] = matrix_to_rotation_6d(ee_rotation_matrix[0:3, 0:3])

        return torch.tensor(rotation_6d_traj)  # (H, 9)
    


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=1/256, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.5, 0, 0.8], [0.0, 0.0, 0.0])
    # Design scene
    
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
 
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    franka_init_pose =  torch.tensor([[-0.5246,  0.3741,  0.7812, -1.9760, -1.2856,  1.6066, -0.0263, 0, 0]], device = sim.device)

    env = Grits(init_pose = franka_init_pose)
    env.run_simulator(sim, scene)



if __name__ == "__main__":
    # run the main function
  
    main()
    # close sim app
    # modify IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py line 658 to shud down the app immediately
    simulation_app.close()
    
