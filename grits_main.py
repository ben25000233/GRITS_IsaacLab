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
from functions.Env_functions import TableTopSceneCfg, Env_functions
from functions.functions import functions

np.random.seed(42)
torch.manual_seed(42)


class Grits():
    def __init__(self, init_pose = None, sim_dt = 1/240):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.spillage_predictor = spillage_predictor()
        self.lfd = LfD()

        config_dir = "config"
        config_file_name = "grits.yaml"
        self.cfg = configparser.get_config(config_dir = config_dir, file_name=config_file_name) 

        self.franka_init_pose = init_pose
    

        # self.gt_front = np.load("./real_cam_pose/front_cam2base.npy")
        self.gt_back = np.load("./real_cam_pose/back_cam2base.npy")

        self.ref_bowl = np.load("./ref_pcd/ref_bowl_pcd.npy")
        self.real_food = np.load("./ref_pcd/real_food.npy")

        self.init_spoon_pcd = np.load("./ref_pcd/ref_spoon_pcd.npy")
        # self.pcd_offset = np.load("./ref_pcd/real_spoon_pcd_offset.npy")

        self.eepose_offset = 0.035

        # self.pcd_offset = np.load("./ref_pcd/temp_offset.npy")

        self.ref_bowl = np.load("./ref_pcd/small_bowl_pcd.npy")
        self.pcd_offset = np.load("./ref_pcd/small_real_tool_offset.npy")

        
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

        self.predict_acc = []
        self.env_functions = Env_functions()


    def get_info(self, robot_entity_cfg = None):


        # front_rgb_image  = self.front_camera.data.output["rgb"][0].cpu().numpy()
        # front_depth_image  = self.front_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        # front_seg_image  = self.front_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        back_rgb_image  = self.back_camera.data.output["rgb"][0].cpu().numpy()
        back_depth_image  = self.back_camera.data.output["distance_to_image_plane"][0].cpu().numpy()
        back_seg_image  = self.back_camera.data.output["semantic_segmentation"][0].cpu().numpy()

        # limage_array = np.load("image_array.npy")[0]

        # print(back_rgb_image.shape)
        # print(limage_array.shape)

        # if np.array_equal(back_rgb_image, limage_array):
        #     print("same img data")
        # else :
        #     print("different img data")
        

        # plt.imshow(back_rgb_image)
        # plt.show()

        # plt.imshow(limage_array)
        # plt.show()

        # simulation_app.close()


        food_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "food", object_id = self.food_semantic_id)
        back_food_world = self.pcd_functions.transform_to_world(food_pcd[:, :3], self.gt_back)
        object_seg = np.full((back_food_world.shape[0], 1), 2)
        back_food_world = np.hstack((back_food_world, object_seg))



        # bowl_pcd = self.pcd_functions.depth_to_point_cloud(back_depth_image[..., 0], back_seg_image[..., 0], object_type = "bowl", object_id = self.bowl_semantic_id)
        # back_bowl_world = self.pcd_functions.transform_to_world(bowl_pcd[:, :3], self.gt_back)
        # object_seg must be 4
        # object_seg = np.full((back_bowl_world.shape[0], 1), 3)
        # back_bowl_world = np.hstack((back_bowl_world, object_seg))


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
        # tool_pcd = self.pcd_functions.depth_to_point_cloud(front_depth_image[..., 0], front_seg_image[..., 0], object_type = "robot", object_id = self.robot_semantic_id)
        # front_tool_world = self.pcd_functions.transform_to_world(tool_pcd[:, :3], self.gt_front)
        # front_tool_world = front_tool_world[front_tool_world[:, 0] > 0.1]
        # front_tool_world = front_tool_world[front_tool_world[:, 2] < 0.2]
        # front_tool_world = self.pcd_functions.align_point_cloud(front_tool_world, target_points = 10000)

        # object_seg = np.full((front_tool_world.shape[0], 1), 5)
        # front_tool_world = np.hstack((front_tool_world, object_seg))
        
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

        back_food_world = self.pcd_functions.align_point_cloud(back_food_world, target_points = 1000)

        mix_all_pcd = np.concatenate(( trans_tool, back_food_world, self.ref_bowl), axis=0)

        mix_all_nor_pcd = self.pcd_functions.nor_pcd(mix_all_pcd)


        # nor_real_pcd = np.load("./ref_pcd/nor_real_pcd.npy")
        # nor_real_pcd[:, 3] = 1
        
        # mix_all_nor_pcd[:, 3] = 2
      
        # check_nor_pcd = np.concatenate(( mix_all_nor_pcd, nor_real_pcd), axis=0)

        # self.pcd_functions.check_pcd_color(check_nor_pcd)
        # simulation_app.close()

        
        # self.pcd_functions.check_pcd_color(mix_all_nor_pcd)
        # simulation_app.close()

    
    
        # self.front_rgb_list[0].append(front_rgb_image)
        # self.front_depth_list[0].append(front_depth_image)
        # self.front_seg_list[0].append(front_seg_image)

        self.back_rgb_list[0].append(back_rgb_image)
        self.back_depth_list[0].append(back_depth_image)
        self.back_seg_list[0].append(back_seg_image)

        self.mix_all_pcd_list[0].append(mix_all_nor_pcd)

    def move_generate(self, guidance_trigger = False):
  
        if len(self.back_rgb_list[0]) == 1:

            # begin_rgb = np.array(self.back_rgb_list[0]).copy()
            # print(np.array(self.back_rgb_list).shape)
            # simulation_app.close()

            # images = [np.array(begin_rgb[0])] * 5
            images = [np.array(self.back_rgb_list[0])[0]] * 5 
            depths = [np.array(self.back_depth_list[0])[0].astype(np.float32)] * 5 
            eepose = [np.array(self.record_ee_pose[0])[0].astype(np.float32)] * 5
            seg_pcd = [np.array(self.mix_all_pcd_list[0])[0]]*3

            
        else:
            images = self.back_rgb_list[0][-5:]  
            depths = self.back_depth_list[0][-5 :]
            eepose = self.record_ee_pose[0][-5:]
            seg_pcd = self.mix_all_pcd_list[0][-3:]
            # self.pcd_functions.check_pcd_color(np.array(seg_pcd[0]))


        image_array = np.array(images)
        depth_array = np.array(depths)
        eepose_array = np.array(eepose)
        seg_pcd_array = np.array(seg_pcd)


        eeposes, n_ori_action, n_pre_action = self.lfd.run_model(image_array, depth_array, eepose_array, seg_pcd_array, guidance_trigger = guidance_trigger)
     

        seg_pcd_array = torch.tensor(seg_pcd_array).to(self.device)


        return eeposes, seg_pcd_array, n_ori_action, n_pre_action

    
    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        """Runs the simulation loop."""
        # Extract scene entities
        # note: we only do this here for readability.
        self.robot = scene["robot"]
        # self.front_camera = scene["front_camera"]
        self.back_camera = scene["back_camera"]
        self.device = sim.device
        self.food_objects = scene["rigid_object"]
        self.backup_objects = scene["backup_object"]

        


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

        # for perturbation
        modify_time = self.cfg.perturbation.time
        add_back = 0
        check_add = 0


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

        # set control rate
        control_rate = 10
        durarion = int(1/sim_dt/control_rate)
        reset_frame = durarion * 10 -2
            
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
   
                if frame_num % durarion == 0:
                    
                    self.get_info(robot_entity_cfg)
            

                    if current_goal_idx % self.action_horizon == 0 and check_add == 0:
                   
                        if self.cfg.guidance == True and current_goal_idx >= 0 :
                            guidance_trigger = True
                        else :
                            guidance_trigger = False
                        action, seg_pcd_array, n_ori_action, n_pre_action = self.move_generate(guidance_trigger = guidance_trigger)
                        # print(n_pre_action)
                        # simulation_app.close()



                        ori_spillage_logic = self.spillage_predictor.validate(n_ori_action, seg_pcd_array)
                        ori_spillage_prob = torch.nn.functional.softmax(ori_spillage_logic[0], dim=-1)[1]

                        pre_spillage_logic = self.spillage_predictor.validate(n_pre_action, seg_pcd_array)
                        pre_spillage_prob = torch.nn.functional.softmax(pre_spillage_logic[0], dim=-1)[1]
                        
                        self.predict_acc.append(pre_spillage_prob.item())

                        print()
                        print(f"original spillage prob : {ori_spillage_prob}")
                        print(f"predicted spillage prob: {pre_spillage_prob}")
                        print()
                        
                        print("current_goal_idx : ", current_goal_idx)
                        self.cal_spillage_scooped(scene = scene, reset = 0)

                    
                    if current_goal_idx < 90 :
                        goal_pose = torch.tensor(action[current_goal_idx % self.action_horizon]).to(self.device)
                    else :
                        break
                    
                    joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    ik_commands[:] = goal_pose
                    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
                    # # reset controller
                    diff_ik_controller.reset()
                    diff_ik_controller.set_command(ik_commands)

                    if self.cfg.perturbation.enable == True : 
                        if current_goal_idx == modify_time and check_add == 0:
                            if self.cfg.perturbation.type == "move" :
                                self._spawn_cubes_at_time(torch.tensor([0], device=self.device), self.food_objects)
                            elif self.cfg.perturbation.type == "add" :
                                self._spawn_cubes_at_time(torch.tensor([0], device=self.device), self.backup_objects)
                            check_add = 1
                        if current_goal_idx == modify_time and check_add == 1:
                            current_goal_idx -= 1
                            add_back += 1
                        if add_back == self.action_horizon:
                            self.cal_spillage_scooped(scene = scene, reset = 1)
                            current_goal_idx += 1
                            add_back += 1
                            check_add = 0

                    # change goal
                    current_goal_idx += 1
                    

                    

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

        r_radius = self.cfg["food_property"]["r_radius"]
        mass = self.cfg["food_property"]["mass"]
        friction = self.cfg["food_property"]["friction"]
        ball_amount = self.cfg["food_property"]["ball_amount"]
        shape = self.cfg["food_property"]["shape"]
        weight = self.cfg.testing.spillage_guided.weight


        with open("./init_setting_for_val/noise_pairs.yaml", "r") as stream:
            noise_cfg = yaml.load(stream, Loader=yaml.FullLoader)

        
        binary_predict_acc = [1 if prob >= 0.5 else 0 for prob in self.predict_acc]
        binary_spillage_amount = [1 if amount > 0 else 0 for amount in self.spillage_amount[0]]


        
        file_name = f"result/logic/fix/size/cube.json"
        try:
            with open(file_name, "r") as json_file:
                spillage_data = json.load(json_file)
        except FileNotFoundError:
            spillage_data = {"r_radius": r_radius,
                             "mass": mass,
                             "ball_amount": ball_amount,
                             "guided_weight": weight,
                             "spillage_scoop": [],
                             "predict_result_list": [],}

   
        # Append the current spillage amount to the array
        spillage_scoop = [sum(self.spillage_amount[0]), self.scooped_amount[0][-1]]
        spillage_data["spillage_scoop"].append(spillage_scoop)
        spillage_data["predict_result_list"].append(binary_predict_acc == binary_spillage_amount)

        # Write the updated array back to the JSON file
        # with open(file_name, "w") as json_file:
        #     json.dump(spillage_data, json_file, indent=4)

        print(f"Spillage data saved: {spillage_data}")

        print(np.array(self.back_rgb_list[0]).shape) # (91, 960, 1280, 3)
        

    def cal_spillage_scooped(self, env_index = 0, reset = 0, scene = None):
        # reset = 1 means record init spillage in experiment setting 

        rigid_object = scene["rigid_object"].data.body_link_state_w
        backup_object = scene["backup_object"].data.body_link_state_w
        rigid_object = torch.cat((rigid_object, backup_object), dim=0)

        y_pose = rigid_object[:,0, 1].to("cpu")
        z_pose = rigid_object[:,0, 2].to("cpu")

        spillage_mask = np.logical_or(z_pose < 0, y_pose > -0.02)
        current_spillage = np.count_nonzero(spillage_mask)

        scoop_mask = np.logical_or(z_pose > 0.12, np.logical_and(z_pose > 0, y_pose > 0))
        scoop_amount = np.count_nonzero(scoop_mask)
        
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


    def _generate_new_ball_positions(self) -> torch.Tensor:
        """
        Generates new origins for the cubes based on the current food properties.
        This is a modified version of define_origins from Env_functions, generating the tensor directly.
        """
        # Load properties (which you save in add_rigid)
        with open("./config/food_info.yaml", "r") as f:
            food_info_cfg = yaml.safe_load(f)
            
        r_radius = food_info_cfg["r_radius"]
        l_radius = food_info_cfg["l_radius"]
        n = food_info_cfg["len"]
        if self.cfg.perturbation.type == "move":
            layer = food_info_cfg["init_ball_amount"]
        else:
            layer = food_info_cfg["backup_ball_amount"]
        self.num_balls = n * n * layer
        
        # Recalculate spacing and offsets based on your Env_functions logic
        spacing = max(r_radius, l_radius) * 2
        
        # Calculate the total number of origins
        num_origins = n * n * layer

        # Initialize a tensor to store all origins
        env_origins = torch.zeros(num_origins, 3, device=self.device)
        
        # Create 2D grid coordinates for the n x n grid in each layer
        xx, yy = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="xy")
        xx = xx.flatten() * spacing - spacing * (n - 1) / 2
        yy = yy.flatten() * spacing - spacing * (n - 1) / 2

        # Noise values are currently in env_functions, but for dynamic spawning, 
        # let's use a simpler fixed offset for now, or sample new noise if desired.
        
        # Base position (where you place the bowl)
        base_x, base_y = 0.575, -0.11 
        # z is set to 0.07 to make balls are inside bowl initially easily
        # z is set to 0.12 for adding new balls above bowl rim
        if self.cfg.perturbation.type == "move":
            base_z = 0.08
        else:
            base_z = 0.12

        # Fill in the coordinates for each layer
        for layer_idx in range(layer):
            start_idx = layer_idx * n * n
            end_idx = start_idx + n * n

            noise_x = random.uniform(-0.03, 0.03)
            noise_y = random.uniform(-0.03, 0.03)

            # Sample new noise for each layer dynamically if needed, 
            # otherwise, keep the original logic for pattern generation.
            
            # Using fixed offsets from your original logic (0.58, -0.12) which matches bowl pos
            env_origins[start_idx:end_idx, 0] = xx + base_x + noise_x
            env_origins[start_idx:end_idx, 1] = yy + base_y
            env_origins[start_idx:end_idx, 2] = layer_idx * spacing + base_z

        return env_origins


    def _spawn_cubes_at_time(self, env_ids: torch.Tensor, food_objects):
        """
        Repositions all cubes in the specified environments.
        """
        # 1. Determine how many cubes are in the collection 
        # (It's num_objects / num_envs, which is all the food in one env)

        
        # 2. Generate the new positions for all cubes (currently only for env 0)
        # Note: We assume only one environment is being reset at a time.
            
        # z is set to 0.07 to make balls are inside bowl initially easily
        # z is set to 0.12 for adding new balls above bowl rim
        n = self.cfg["food_property"]["len"]
        if self.cfg["perturbation"]["type"] == "move":
            layer = self.cfg["food_property"]["init_ball_amount"]
            init_pose = (0.58, -0.12, 0.08)
        else:
            layer = self.cfg["food_property"]["backup_ball_amount"]
            init_pose = (0.58, -0.12, 0.12)
        num_balls = n * n * layer
        new_positions = self.env_functions.define_origins(
            n = n, 
            layer = layer, 
            spacing = max(self.cfg["food_property"]["r_radius"], self.cfg["food_property"]["l_radius"]) * 2, 
            init_pose = init_pose,
            )

        
        # 3. Create the state tensor for the cubes in the target environments
        # State: [num_envs * num_objects_per_env, 13]
        state_dim = 13
        current_state = torch.zeros((len(env_ids) * num_balls, state_dim), device=self.device)
        
        # 4. Populate the state tensor with the new positions (and default orientation/velocity)
        for i, env_id in enumerate(env_ids):
            start_idx = i * num_balls
            end_idx = start_idx + num_balls
            
            # Set position (indices 0, 1, 2)
            current_state[start_idx:end_idx, 0:3] = torch.tensor(new_positions, device=self.device)
            
            # Set quaternion to identity (wxyz, indices 3, 4, 5, 6)
            current_state[start_idx:end_idx, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device) 
            
            # Set linear and angular velocities to zero (indices 7-12)
            current_state[start_idx:end_idx, 7:] = 0.0

        # 5. Write the new state back to the simulation
        body_indices_to_update = food_objects.body_names[env_ids]
        food_objects.write_root_state_to_sim(current_state)
        
    
    # def qua_to_rotation_6d(self, ee_traj):
        
    #     ee_traj = torch.tensor(np.array(ee_traj), dtype=torch.float32)

    #     rotation_6d_traj = torch.zeros((ee_traj.shape[0], 9))  # (H, 9)

        
    #     for i in range(ee_traj.shape[0]):
    #         ee = ee_traj[i]
    #         quaternion = ee[3:]  # (qw, qx, qy, qz)

    #         # Convert quaternion to rotation matrix
    #         ee_rotation_matrix = quaternion_to_matrix(quaternion)

    #         rotation_6d_traj[i, :3] = ee[:3]
    #         rotation_6d_traj[i, 3:] = matrix_to_rotation_6d(ee_rotation_matrix[0:3, 0:3])

    #     return torch.tensor(rotation_6d_traj)  # (H, 9)
    


def main():
    """Main function."""
    # Load kit helper
    sim_dt = 1 / 120

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
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
    
