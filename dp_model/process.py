from pyconfigparser import configparser
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser
from model import RotationTransformer
from tqdm import tqdm
import os

    
# diffusion_policy/diffusion_policy/model/common/normalizer.py
def _normalize(data, input_max, input_min, input_mean):
    ranges = input_max - input_min
    data_normalize = np.zeros_like(data)
    for i in range(3):
        if ranges[i] < 1e-4:
            # If variance is small, shift to zero-mean without scaling
            data_normalize[:, i] = data[:, i] - input_mean[i]
        else:
            # Scale to [-1, 1] range
            data_normalize[:, i] = -1 + 2 * (data[:, i] - input_min[i]) / ranges[i]    
    data_normalize[:, 3:] = data[:, 3:]
    return data_normalize

# origin:[960, 1280] -> resize:[240, 320] (h, w)
def rgb_transform(img):
    return transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.ToTensor(),
    ])(img)

def depth_transform(img):
    return transforms.Compose([
        transforms.Resize([240, 320]),
        transforms.ToTensor(),
    ])(img)

def main():

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="template.yaml")
    parser.add_argument("-i", "--in_dir", default="dataset_01_13")
    parser.add_argument("-o", "--out_dir", default="split_dataset_01_13")
    args = parser.parse_args()
    cfg = configparser.get_config(file_name=args.config)

    # ========================================= #
    # T=16, To=5, Ta=8
    # |o|o|o|o|o|
    # | | | | |a|a|a|a|a|a|a|a|
    # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|
    # ========================================= #

    dir_path = args.in_dir
    out_dir = args.out_dir
    food_item = cfg.data.food_item
    
    # food category
    num_all = 0
    rotation_transformer = RotationTransformer('quaternion', 'rotation_6d')
    
    ee_all = []
    for i in range(len(food_item)):
        for j in range(cfg.data.trial_num):
            ee_pose = np.load(dir_path + "/" + food_item[i] + "_{}/ee_pose_qua.npy".format(str(j+1).zfill(2)))
            ee_all.append(ee_pose)

    ee_all = np.asarray(ee_all)
    ee_all = np.reshape(ee_all, (ee_all.shape[0]*ee_all.shape[1],ee_all.shape[2]))
    input_max = np.max(ee_all, axis=0)
    input_min = np.min(ee_all, axis=0)
    input_mean = np.mean(ee_all, axis=0)
    input_range = torch.from_numpy(np.concatenate((
        np.expand_dims(input_max, 0), 
        np.expand_dims(input_min, 0), 
        np.expand_dims(input_mean, 0)), 0))
    torch.save(input_range, out_dir + "/input_range.pt")
    print("input_max= ", input_max)
    print("input_min= ", input_min)
    print("input_mean= ", input_mean)

    for i in tqdm(range(len(food_item))):
        for j in range(cfg.data.trial_num):
            print(food_item[i] + '_' + str(j+1).zfill(2))

            rgb_front_orin = np.load(dir_path + "/" + food_item[i] + "_{}/front_rgb.npy".format(str(j+1).zfill(2)))
            depth_front_orin = np.load(dir_path + "/" + food_item[i] + "_{}/front_depth.npy".format(str(j+1).zfill(2)))

            rgb_back_orin = np.load(dir_path + "/" + food_item[i] + "_{}/back_rgb.npy".format(str(j+1).zfill(2)))
            depth_back_orin = np.load(dir_path + "/" + food_item[i] + "_{}/back_depth.npy".format(str(j+1).zfill(2)))

            ee_pose_orin = np.load(dir_path + "/" + food_item[i] + "_{}/ee_pose_qua.npy".format(str(j+1).zfill(2)))

            # ------------crop----------
            rgb_front = rgb_front_orin[10:230, :800, 80:]
            depth_front = depth_front_orin[10:230, :800, 80:]
            rgb_back = rgb_back_orin[10:230, :800, 80:]
            depth_back = depth_back_orin[10:230, :800, 80:]
            ee_pose = ee_pose_orin[10:230]

            assert rgb_front.shape[0]==220
            assert depth_front.shape[0]==220
            assert rgb_back.shape[0]==220
            assert depth_back.shape[0]==220
            assert ee_pose.shape[0]==220

            # convert qua to rotation 6D
            ee_pose_position = ee_pose[:, :3]
            ee_pose_rotation = rotation_transformer.forward(ee_pose[:, 3:])
            ee_pose_6d = np.concatenate((ee_pose_position, ee_pose_rotation), -1) # [:,9]

            # add To-1 frame before first frame
            To = cfg.n_obs_steps
            for _ in range(To-1):
                rgb_front = np.concatenate((np.expand_dims(rgb_front[0], 0), rgb_front), 0)
                depth_front = np.concatenate((np.expand_dims(depth_front[0], 0), depth_front), 0)

                rgb_back = np.concatenate((np.expand_dims(rgb_back[0], 0), rgb_back), 0)
                depth_back = np.concatenate((np.expand_dims(depth_back[0], 0), depth_back), 0)

                ee_pose_6d = np.concatenate((np.expand_dims(ee_pose_6d[0], 0), ee_pose_6d), 0)               

            assert rgb_front.shape[0]==220+To-1
            assert depth_front.shape[0]==220+To-1
            assert rgb_back.shape[0]==220+To-1
            assert depth_back.shape[0]==220+To-1
            assert ee_pose_6d.shape[0]==220+To-1

            T = cfg.horizon
            t=0
            while t<ee_pose.shape[0]-T: # 254-16=238
                    
                rgb_front_tslice = rgb_front[t:t+T] # [16]
                depth_front_tslice = depth_front[t:t+T] # [16,]

                rgb_back_tslice = rgb_back[t:t+T] # [16]
                depth_back_tslice = depth_back[t:t+T] # [16,]

                ee_pose_6d_tslice = ee_pose_6d[t:t+T] # [16, 7]
                # action normalize to [-1,1], and transfer to tensor
                ee_6d_tslice_normalize = _normalize(ee_pose_6d_tslice, input_max, input_min, input_mean)

                ob_front_tensor = []
                ob_back_tensor = []
                for tt in range(T):
                    # observation transform
                    
                    rgb_PIL =  Image.fromarray(rgb_front_tslice[tt].astype('uint8'), 'RGB')
                    rgb_front_tt = rgb_transform(rgb_PIL)
                    # front depth
                    depth_PIL = depth_front_tslice[tt].astype(np.float32)
                    depth_PIL = depth_PIL / np.max(depth_PIL)
                    depth_front_tt = depth_transform(Image.fromarray(depth_PIL))
                    # concat
                    ob_front_tensor.append(torch.cat((rgb_front_tt, depth_front_tt), 0))    

                    # back rgb
                    rgb_PIL =  Image.fromarray(rgb_back_tslice[tt].astype('uint8'), 'RGB')
                    rgb_back_tt = rgb_transform(rgb_PIL)
                    # back depth
                    depth_PIL = depth_back_tslice[tt].astype(np.float32)
                    depth_PIL = depth_PIL / np.max(depth_PIL)
                    depth_back_tt = depth_transform(Image.fromarray(depth_PIL))
                    # concat
                    ob_back_tensor.append(torch.cat((rgb_back_tt, depth_back_tt), 0))         
                
                # save tensor
                ee_6d_tensor = torch.from_numpy(ee_6d_tslice_normalize) # (16, 9)
                ob_front_tensor = torch.stack(ob_front_tensor, dim=0)
                ob_back_tensor = torch.stack(ob_back_tensor, dim=0)

                assert ob_front_tensor.shape[0]==T
                assert ob_back_tensor.shape[0]==T
                assert ee_6d_tensor.shape[0]==T

                if not os.path.exists(f"{out_dir}/ob_front"):
                    os.makedirs(f"{out_dir}/ob_front")
                if not os.path.exists(f"{out_dir}/ob_back"):
                    os.makedirs(f"{out_dir}/ob_back")
                if not os.path.exists(f"{out_dir}/traj"):
                    os.makedirs(f"{out_dir}/traj")

                torch.save(ob_front_tensor, out_dir+'/ob_front/ob_front_{}.pt'.format(str(num_all).zfill(5)))
                torch.save(ob_back_tensor, out_dir+'/ob_back/ob_back_{}.pt'.format(str(num_all).zfill(5)))
                torch.save(ee_6d_tensor, out_dir+'/traj/traj_{}.pt'.format(str(num_all).zfill(5)))
                num_all+=1
                # print(num_all) # one trial = 234 .pt file   

                t+=1 
          

if __name__ == '__main__':
    main()
