import os
import sys
import numpy as np
import vedo
import cv2
import json
from data_loader import load_volume_from_dir

def get_bndbox(src_volume_size, affine_matrix):
    a = affine_matrix[:3,:3]
    b = affine_matrix[:3,3]
    #xyz
    z, y, x = src_volume_size

    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])

    new_points = np.zeros_like(points)
    for i, point in enumerate(points):
        new_point = np.floor(np.dot(a,point) + b.T)
        new_points[i]= new_point

    x_max = max(new_points[:,0])
    y_max = max(new_points[:,1])
    z_max = max(new_points[:,2])

    x_min = min(new_points[:,0])
    y_min = min(new_points[:,1])
    z_min = min(new_points[:,2])

    delta_x = 0
    delta_y = 0
    delta_z = 0
    if x_min < 0 or y_min < 0 or z_min < 0:
        if x_min < 0 :
            delta_x = -x_min+np.floor(0.005*src_volume_size[0])
        if y_min < 0 :
            delta_y = -y_min+np.floor(0.005*src_volume_size[1])
        if z_min < 0 :
            delta_z = -z_min+np.floor(0.005*src_volume_size[2])

    return np.array([x_max, y_max , z_max]), np.array([delta_x,delta_y,delta_z]) 


def add_gaussian_smoothing(image, sigma):
    smoothed = np.zeros_like(image, dtype=np.float32)
    
    # Apply 2D Gaussian filter to each slice along Z-axis
    for i in range(image.shape[2]):  
        smoothed[:, :, i] = cv2.GaussianBlur(image[:, :, i], (0, 0), sigma)
    
    return smoothed

def add_gaussian_noise(image, noise_level, seed):
    rng = np.random.default_rng(seed) #Initialisation of random 

    flatimage = image.flatten()
    threshold = np.percentile(flatimage, 95)  #Ignore top 5% of values
    filtered_flatimage = flatimage[flatimage <= threshold]

    if filtered_flatimage.size > 0:
        max_value = np.max(filtered_flatimage)
    else:
        print("Empty filterred array")
        max_value = 0
    sigma = noise_level * max_value  #Adjust sigma based on max intensity
    print(f"noise level {noise_level}     max_value {max_value}")
    noise = rng.normal(0, sigma, image.shape) #Gaussian noise
    noisy_image = image + np.astype(noise,np.float32) 
    return noisy_image

def make_sample(src_volume, matrix, transformed_volumes_path,noise_level = 0.0,gaussian_sigma = 0.0, randomseed = 42, flag = False):
    linear_transform = vedo.LinearTransform()
    linear_transform.matrix = matrix

    new_volume_shape, origin_offset = np.array(get_bndbox(src_volume.shape, matrix))
    print(f'new volume shape (x, y, z) = {new_volume_shape}')
    
    vedo_volume = vedo.Volume(src_volume)
    vedo_volume.permute_axes(2,1,0)

    try:
        vedo_volume.apply_transform(linear_transform, fit = True, interpolation = 'linear')
    except:
        pass

    empty_arr = np.zeros(new_volume_shape)
    vedo_volume_new = vedo.Volume(empty_arr, origin = [0,0,0], dims = empty_arr.shape)
    vedo_volume_new.resample_data_from(vedo_volume)
    vedo_volume = vedo_volume_new
    del vedo_volume_new
    vedo_volume.permute_axes(2,1,0)
    transformed_volume = vedo_volume.tonumpy().astype(np.float32)
    #tools.show_3d_volume(save_volume)
    del vedo_volume

    if gaussian_sigma == 0.0:
        smothed_volume = transformed_volume
    else:
        smothed_volume = add_gaussian_smoothing(transformed_volume, gaussian_sigma)
    del transformed_volume
    
    if noise_level == 0.0:
        save_volume = smothed_volume 
    else:
        save_volume = add_gaussian_noise(smothed_volume, noise_level, randomseed)
    del smothed_volume

    for i, v_slice in enumerate(save_volume):
        save_path = os.path.join(transformed_volumes_path, str("%04d"%i) + '.tif')#slice_{i:05d}
        cv2.imwrite(save_path, v_slice)


if __name__ == "__main__":

    src_volume_path          = sys.argv[1]
    info_path                = sys.argv[2]
    transformed_volumes_path = sys.argv[3]
    minimize_padding         = sys.argv[4]
    x_shape                  = sys.argv[5]
    y_shape                  = sys.argv[6]
    z_shape                  = sys.argv[7]
     
    matrix_path = info_path
    src_volume = load_volume_from_dir(src_volume_path)
    with open(matrix_path, 'r', encoding='UTF-8') as json_file:
        matrix = np.array(json.load(json_file)["matrix"])

    linear_transform = vedo.LinearTransform()
    linear_transform.matrix = matrix

    new_volume_shape, origin_offset = get_bndbox(src_volume.shape, matrix)
    print(f'new volume shape (x, y, z) = {new_volume_shape}, {origin_offset}')
    if minimize_padding == "True":
        new_volume_shape = [int(x_shape), int(y_shape), int(z_shape)]
        origin_offset = [0,0,0]
    else:
        print(f'origin offset (x, y, z) = {origin_offset}')
        if origin_offset[0] > 0 or origin_offset[1] > 0 or origin_offset[2] > 0:
            origin_offset = origin_offset.astype(np.int32)
            new_volume_shape = new_volume_shape+origin_offset
        with open(f'{os.path.join(info_path.rsplit("/",1)[0],"origin_negative_offset.json")}', 'w', encoding='UTF-8') as f:
            json.dump({"origin_offset":origin_offset.astype(np.int32).tolist()}, f)
            print(f'Expanded to negative areas new volume shape (x, y, z) = {new_volume_shape}')

    vedo_volume = vedo.Volume(src_volume)
    del src_volume
    vedo_volume.permute_axes(2,1,0)
    try:
        vedo_volume.apply_transform(linear_transform, fit = True, interpolation = 'cubic')
    except:
        pass

    #memory economy
    slices_per_chunk = 30
    n_chunks = new_volume_shape[2]//slices_per_chunk
    if n_chunks==0:
        n_chunks=1
    for i in range(n_chunks):
        if i == n_chunks - 1:
            chunk_volume = vedo.Volume( origin = [-origin_offset[0],-origin_offset[1],-origin_offset[2]+i*slices_per_chunk], dims = np.array([new_volume_shape[0],new_volume_shape[1],new_volume_shape[2]-i*slices_per_chunk]))
        else:
            chunk_volume = vedo.Volume(origin = [-origin_offset[0],-origin_offset[1],-origin_offset[2]+i*slices_per_chunk], dims = np.array([new_volume_shape[0],new_volume_shape[1],slices_per_chunk]))
        chunk_volume.resample_data_from(vedo_volume)
        chunk_volume.permute_axes(2,1,0)
        save_volume = chunk_volume.tonumpy().astype(np.float32)
        del chunk_volume
        if i%5 == 0:
            print(f"{i}_new_cunk_ok",'\r')
        for j, v_slice in enumerate(save_volume):
            save_path = os.path.join(transformed_volumes_path, str("%04d"%(i*slices_per_chunk+j)) + '.tif')#slice_{i:05d}
            cv2.imwrite(save_path, v_slice)
        del save_volume
