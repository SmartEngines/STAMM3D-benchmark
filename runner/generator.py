"""
Transformation generator for volumes. 
"""
import os
import sys
import json
from PIL import Image
from vedo import LinearTransform
import numpy as np
from joblib import Parallel, delayed
import sample_creator
import shutil
from data_loader import load_volume_from_dir

def make_folder(folder_path, delete_existing = True):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif delete_existing: 
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def get_affine_matrix(scale, angles, offsets):
    """
    Returns an affine matrix given angles and offsets.
    """
    angle_z, angle_y, angle_x = angles
    offset_x, offset_y, offset_z = offsets
    linear_transform = LinearTransform()
    linear_transform.scale(scale).rotate_z(angle_z).rotate_y(angle_y).rotate_x(angle_x)
    linear_transform.translate([offset_x, offset_y, offset_z])
    return np.array(linear_transform.matrix.tolist())

def get_markup_shape(path):
    """
    Returns markup shape.
    """
    files = os.listdir(path)
    images_format = '.' + files[0].split('.')[1]
    tif_files = [os.path.join(path, x) for x in files if images_format in x]
    size_z = len(tif_files)
    size_y = np.array(Image.open(tif_files[0])).shape[0]
    size_x = np.array(Image.open(tif_files[0])).shape[1]
    return np.array([size_x, size_y, size_z])

def get_min_offsets(volume_shape, affine_matrix):
    """
    Returns corner volume coordinate after trasformation, which has smallest values.
    """
    a = affine_matrix[:3,:3]
    b = affine_matrix[:3,3]
    x, y, z = volume_shape
    points = np.array([ [0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])

    new_points = np.zeros_like(points)
    for i, point in enumerate(points):
        new_point = np.floor(np.dot(a,point) + b.T)
        new_points[i]= new_point

    x_min = min(new_points[:,0])
    y_min = min(new_points[:,1])
    z_min = min(new_points[:,2])

    return [x_min, y_min, z_min]

def check_affine_matrix(volume_shape, matrix):
    """
    Checks that the volume is not in the negative region after applying the transformation.
    """
    x_min, y_min, z_min = get_min_offsets(volume_shape, matrix)
    delta_x = 0
    delta_y = 0
    delta_z = 0
    if x_min < 0 or y_min < 0 or z_min < 0:
        print("Part of the volume is in the negative area,")
        print("change the offsets of affine transfromation.")
        print(f'Current volume origin position is {x_min, y_min, z_min}.')
        if x_min < 0 :
            delta_x = -x_min+np.floor(0.005*volume_shape[0])
        if y_min < 0 :
            delta_y = -y_min+np.floor(0.005*volume_shape[1])
        if z_min < 0 :
            delta_z = -z_min+np.floor(0.005*volume_shape[2])
    else:
        print(f'Current volume origin position is {x_min, y_min, z_min}.')
        [x_min, y_min, z_min]=[0.0,0.0,0.0]
    return [delta_x,delta_y,delta_z]


def get_coord_center_mass_slice(volume, z):
    """
    Calculating for center of mass.
    """
    numerator = [0,0,0]
    denominator = 0
    if z%20 == 0:
        print(f'z = {z}/{volume.shape[0]}')
    for y in range(volume.shape[1]):
        for x in range(volume.shape[2]):
            numerator += volume[z][y][x] * (np.array([x,y,z]))# - point)
            denominator += volume[z][y][x]
    return [numerator, denominator]

def get_coord_center_mass(volume):
    """
    Calculating volume center of mass.
    """
    print('Calculating coordinates of center of mass:')
    res = Parallel(n_jobs=10)(delayed(get_coord_center_mass_slice)(volume, z)
                                             for z in range(volume.shape[0]))
    numerator = [0,0,0]
    denominator = 0
    for obj in res:
        numerator += obj[0]
        denominator += obj[1]
    print(f'r_C = {numerator/denominator}')
    return numerator/denominator
    
def generate(center_of_rotation_in_origin, volume_params, save_path, markup_path):
    """
    Converts markup volume according to the transformations in the config
    and saves.
    """
    markup_volume_shape = get_markup_shape(markup_path)
    if center_of_rotation_in_origin is False:
        center = markup_volume_shape/2.0
    else:
        center = [0,0,0]
    #getting uncorrected  target matrix for volume with [0.0.0] origin
    matrix = get_affine_matrix(
        volume_params["scale"],
        [volume_params["rotation_0z_deg"],
         volume_params["rotation_0y_deg"],
         volume_params["rotation_0x_deg"]],
        [volume_params["offset_x"],
         volume_params["offset_y"],
         volume_params["offset_z"]])
    matrix[:3,3] = matrix[:3,3]+center-matrix[:3,:3]@center
    #saving matrix for target transform
    volume_params["target_transformation"]=matrix.tolist()
    #calculating offset for markup and adding it to matrix
    markup_offset = check_affine_matrix(markup_volume_shape, matrix)
    if markup_offset[0] > 0 or markup_offset[1] > 0 or markup_offset[2] > 0:
        matrix[:3,3]=matrix[:3,3]+markup_offset
        print(f'Affine matrix = \n{matrix}')
    #saving matrix for volume creation
    volume_params["transformation"]=matrix.tolist()
    #addind markup offset to transform information json
    volume_params["markup_offset"] = markup_offset
    with open(f'{save_path}/tr_gen.json', 'w', encoding='UTF-8') as json_file:
        json.dump(volume_params,json_file,indent=4)

    src_volume = load_volume_from_dir(markup_path)
    if volume_params["varying_random_noise"] is True:
        volume_params["general_random_seed"] += 1
        rng = np.random.default_rng(volume_params["general_random_seed"])
        randomseed = rng.integers(1, 1000001)
    else:
        randomseed = volume_params["general_random_seed"]
    sample_creator.make_sample(src_volume,
                               matrix, save_path,
                               volume_params["noise_level"],
                               volume_params["gaussian_sigma"],
                               randomseed)

if __name__ == '__main__':
    config_path = sys.argv[1].replace(os.sep, '/')
    with open(config_path, 'r', encoding='UTF-8') as json_file:
        config = json.load(json_file)
    
    data_folder = config["path_to_data"]
    group_list = config["testing_group_list"]
    transformed_volumes_path = os.path.join(config_path.rsplit("/",maxsplit=1)[0],"transformed_volumes/")
    make_folder(transformed_volumes_path)
    for group_n, gr_params in enumerate(group_list):
        if gr_params["skip_current_group"]:
            continue
        testing_package = gr_params["testing_package"]
        markup_path = os.path.join(data_folder, "markups", testing_package)
        group_path = os.path.join(config_path.rsplit("/",maxsplit=1)[0],
                                  "transformed_volumes",
                                  f'gr{group_n}_d{gr_params["gr_dimention"]}/')
        make_folder(group_path)

        print(f"Group #{group_n}")
        if gr_params["gr_dimention"] == 2:
            v1_params = gr_params["variation1"]
            v2_params = gr_params["variation2"]
            v1_values = np.linspace(v1_params["left_border"],
                                    v1_params["right_border"],
                                    v1_params["points_amount"])
            v2_values = np.linspace(v2_params["left_border"],
                                    v2_params["right_border"],
                                    v2_params["points_amount"])
            for j in range(v1_params["points_amount"]):
                make_folder(os.path.join(group_path,(v1_params["param_name"]+f"_{j}/")))
                for k in range(v2_params["points_amount"]):
                    print("--------------------------------------")
                    save_path = os.path.join(group_path,
                                            (v1_params["param_name"]+f"_{j}"),
                                            (v2_params["param_name"]+f"_{k}/"))
                    make_folder(save_path)
                    #creating parameters for current volume
                    volume_params = dict(list(gr_params.items()))
                    volume_params[v1_params["param_name"]] = v1_values[j]
                    volume_params[v2_params["param_name"]] = v2_values[k]
                    generate(gr_params["center_of_rotation_in_origin"],
                             volume_params,
                             save_path,
                             markup_path)
                    print("--------------------------------------")

        elif gr_params["gr_dimention"] == 1:
            v1_params = gr_params["variation1"]
            v1_values = np.linspace(v1_params["left_border"],
                                    v1_params["right_border"],
                                    v1_params["points_amount"])
            for j in range(v1_params["points_amount"]):
                print("--------------------------------------")
                save_path = os.path.join(group_path,(v1_params["param_name"]+f"_{j}/"))
                make_folder(save_path)
                #creating parameters for current volume
                volume_params = dict(list(gr_params.items()))
                volume_params[v1_params["param_name"]] = v1_values[j]
                generate(gr_params["center_of_rotation_in_origin"],
                             volume_params,
                             save_path,
                             markup_path)
                print("--------------------------------------")

        elif gr_params["gr_dimention"] == 0:
            print("--------------------------------------")
            save_path = group_path
            #creating parameters for current volume
            volume_params = dict(list(gr_params.items()))
            generate(gr_params["center_of_rotation_in_origin"],
                             volume_params,
                             save_path,
                             markup_path)
            print("--------------------------------------")