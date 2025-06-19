import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as Rot
import random
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    DeformableObject, 
    DeformableObjectCfg,
)
import tqdm
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab_asset.spoon_franka import SCOOP_FRANKA_CFG 
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


class Env_functions():
    def __init__(self):
        pass
    def add_soft(self): 

        r_radius = round(random.uniform(0.0025, 0.01), 4)
        l_radius = round(random.uniform(0.0025, 0.01), 4)
        mass = round(random.uniform(0.0001, 0.005), 4)
        friction = round(random.uniform(0, 1),2)
        max_num = int(256/pow(2, (r_radius - 0.0025)*1000)) 
        amount = random.randint(1, max(1, max_num))+2


        # youngs_modulus = round(random.uniform(0.0001, 0.005), 4)

        soft_origins = self.define_origins(n = 4, layer = amount, spacing=max(r_radius, l_radius) * 2.1)


        # Define deformable material properties (required for soft bodies)
        soft_material = sim_utils.DeformableBodyMaterialCfg(
            youngs_modulus=1e5,
            poissons_ratio=0.4,
            density=None,  # Optional
            elasticity_damping=0.00001,
            dynamic_friction = friction,
        )

        # ----use predefine Sphere shape 
        cfg_sphere = sim_utils.MeshSphereCfg(
            radius=r_radius,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = soft_material,
        )

        cfg_cube = sim_utils.MeshCuboidCfg(
            size=(r_radius * 2, r_radius * 2, l_radius * 2),
            # hight=l_radius * 2,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = soft_material,
        )

        cfg_cone = sim_utils.MeshConeCfg(
            radius=r_radius,
            height=l_radius* 2,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = soft_material,
        )

        cfg_cylinder = sim_utils.MeshCylinderCfg(
            radius=r_radius,
            height=l_radius* 2,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = soft_material,

        )


        
        '''
        # ----use own usd 
        # Define MeshCfg for the soft body
        soft_body_cfg = sim_utils.MeshCfg(
            visual_material=sim_utils.PreviewSurfaceCfg(),  # Assign visual material 
            physics_material=soft_material,  # Assign soft body physics properties
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0),  # Deformable physics properties
        )

        # import usd file 
        str_cfg = sim_utils.UsdFileCfg(
                usd_path=str_usd_path, 
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
                scale = (0.1, 0.1, 0.1)
                )
        '''

        # ---- determine which cfg file be used (cfg_sphere / str_cfg)
        shapes = [cfg_sphere, cfg_cube, cfg_cone, cfg_cylinder]
        obj_cfg = random.choice(shapes)
        # obj_cfg = cfg_cylinder

        obj_cfg.semantic_tags = [("class", "food")]


        for idx, origin in tqdm.tqdm(enumerate(soft_origins), total=len(soft_origins)):
            obj_cfg.func(f"/World/soft/Object{idx:02d}", obj_cfg, translation=origin)
            

        soft_cfg = DeformableObjectCfg(
            prim_path=f"/World/soft/Object.*",
            spawn=None,
            init_state=DeformableObjectCfg.InitialStateCfg(),
            debug_vis=True,
        )

        return soft_cfg

    def add_rigid(self): 
    
        mass = 0.1
        friction = 0.5
        

        ball_amount = 10
        r_radius = 0.003


        l_radius = 0.009
        rigid_origins = self.define_origins(n = 4, layer = ball_amount, spacing=max(r_radius, l_radius) * 2)
        # str_usd_path ="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/str.usd"


        cfg_sphere = sim_utils.SphereCfg(
            radius = r_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
        )

        cfg_cube = sim_utils.CuboidCfg(
            size=(r_radius*2, r_radius*2, l_radius*2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
        )

        cfg_cone = sim_utils.ConeCfg(
            radius=r_radius,
            height=l_radius*2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
        )

        cfg_cylinder = sim_utils.CylinderCfg(
            radius=r_radius,
            height=l_radius*2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction),
        )


        # str_cfg = sim_utils.UsdFileCfg(
        #     usd_path=str_usd_path, 
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #     # mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        #     # collision_props=sim_utils.CollisionPropertiesCfg(),
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        #     scale = (0.1, 0.1, 0.1), 
        # )

        shapes = [cfg_sphere, cfg_cube, cfg_cone, cfg_cylinder]
        obj_cfg = random.choice(shapes)
        obj_cfg = cfg_sphere

        obj_cfg.semantic_tags = [("class", "food")]


        for idx, origin in tqdm.tqdm(enumerate(rigid_origins), total=len(rigid_origins)):
            obj_cfg.func(f"/World/rigid/Object{idx:02d}", obj_cfg, translation=origin)

            

        rigid_cfg = RigidObjectCfg(
            prim_path=f"/World/rigid/Object.*",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(),
            # debug_vis=True,
        )

        # return rigid_cfg, food_info
        return rigid_cfg

    def add_camera(self, cam_type = "front"):

        front_cam_pose = np.load("./real_cam_pose/front_cam2base.npy")
        front_cam_pos = front_cam_pose[0:3, 3]
        front_cam_rot = Rot.from_matrix(front_cam_pose[0:3, 0:3]).as_quat()

        back_cam_pose = np.load("./real_cam_pose/back_cam2base.npy")
        back_cam_pos = back_cam_pose[0:3, 3]
        back_cam_rot = Rot.from_matrix(back_cam_pose[0:3, 0:3]).as_quat()

        focal_length = 16.6

        if cam_type == "front":
            cam_pos = (front_cam_pos[0], front_cam_pos[1], front_cam_pos[2])
            cam_rot = (front_cam_rot[3], front_cam_rot[0], front_cam_rot[1], front_cam_rot[2])

            camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.1,
            height=960,
            width=1280,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "normals",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ],
            colorize_semantic_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
        )
        else:
            cam_pos = (back_cam_pos[0], back_cam_pos[1], back_cam_pos[2])
            cam_rot = (back_cam_rot[3], back_cam_rot[0], back_cam_rot[1], back_cam_rot[2])

            camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/back_cam",
            update_period=0.1,
            height=960,
            width=1280,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "normals",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ],
            colorize_semantic_segmentation=False,
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                # focal_length=19.6, focus_distance=400.0, horizontal_aperture=20.955, 
                focal_length=focal_length, focus_distance=400.0, horizontal_aperture=20.955, 
            ),

            offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
        )



        return camera

    def define_origins(self, n: int, layer: int, spacing: float) -> list[list[float]]:
        """
        Defines the origins of a 3D grid with n * n particles per layer and m layers stacked along the z-axis.

        Args:
            n (int): The number of particles per row/column in each layer (n * n grid).
            layer (int): The number of layers stacked along the z-axis.
            spacing (float): The spacing between particles in the grid and between layers.

        Returns:
            list[list[float]]: A list of origins, where each origin is a 3D coordinate [x, y, z].
        """
        # Calculate the total number of origins
        num_origins = n * n * layer

        # Initialize a tensor to store all origins
        env_origins = torch.zeros(num_origins, 3)

        # Create 2D grid coordinates for the n x n grid in each layer
        xx, yy = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="xy")
        xx = xx.flatten() * spacing - spacing * (n - 1) / 2
        yy = yy.flatten() * spacing - spacing * (n - 1) / 2

        

        # Fill in the coordinates for each layer
        for layer in range(layer):
            noise = round(random.uniform(0.002, 0.01), 4)
            start_idx = layer * n * n
            end_idx = start_idx + n * n

            # Set x, y, and z coordinates for this layer
            env_origins[start_idx:end_idx, 0] = xx + 0.59 + noise
            env_origins[start_idx:end_idx, 1] = yy - 0.11 + noise
            env_origins[start_idx:end_idx, 2] = layer * spacing + 0.1

        # Convert the origins to a list of lists and return
        return env_origins.tolist()
    

env_functions = Env_functions()

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
                    size=(2, 2, 1),
                    # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # articulation


    robot = SCOOP_FRANKA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # soft body
    # soft_object = add_soft()

    # rigid_object = add_rigid()
    rigid_object = env_functions.add_rigid()
    


    # front_camera = add_camera("front")
    # back_camera = add_camera("back")

    front_camera = env_functions.add_camera("front")
    back_camera = env_functions.add_camera("back")

    

    bowl = AssetBaseCfg(
        prim_path="/World/envs/env_.*/bowl",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./isaaclab_asset/USD/tool/bowl/bowl.usd", 
            # usd_path="/home/hcis-s22/benyang/IsaacLab/source/isaaclab_assets/data/tool/spoon/spoon.usd", 
            scale=(1, 1, 1), 
            # making a rigid object static
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset = 0.001, rest_offset = 0.0001),
            semantic_tags = [("class", "bowl"), ("id", "2")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.575, -0.11, 0.02)),
    )
