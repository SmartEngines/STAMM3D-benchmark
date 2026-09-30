import sys
import json
import torch 
import os
import numpy as np
import torchio as tio
import subprocess
import matplotlib.pyplot as plt
import tqdm
import cv2
from PIL import Image
import shutil
from data_loader import load_volume_from_dir

# Нахождение ближайших и похожих
def make_folder(folder_path, delete_existing = True):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif delete_existing: 
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def sopostavlenie_create_and_sort_map(points_real, points_real_max):
    k = 0
    point_map = {}
    for i, p in enumerate(tqdm.tqdm(points_real)):
        for j, p_max in enumerate(points_real_max):
            s = np.linalg.norm(p[4:] - p_max[4:])#euclidean_distance(p[4:], p_max[4:])
            ### was:
            # s = euclidean_distance(p[3:], p_max[3:])
            k += 1
            point_map[k] = {"p": i, "p_max": j, "distance": s}
    
    print('Size of point map: ',len(point_map))
    return point_map

def sort_point_map(point_map):
    sorted_point_map = dict(sorted(tqdm.tqdm(point_map.items()), key=lambda item: item[1]["distance"]))
    print('Size of sorted point map: ',len(sorted_point_map))
    return sorted_point_map

def delete_bad_pairs(sorted_point_map, points_real, points_real_max):
    points_real_sort = []
    points_real_max_sort = []
    values_p = set()
    values_p_max = set()
    
    for value in tqdm.tqdm(sorted_point_map.items()):
        p = int(value[1]['p'])
        p_max = int(value[1]['p_max'])
        if not (p in values_p or p_max in values_p_max):
            points_real_sort.append(points_real[p][:3])
            points_real_max_sort.append(points_real_max[p_max][:3])
            values_p.add(p)
            values_p_max.add(p_max)
    print('Number of comparisons for Vol1: ', len(points_real_sort))
    print('Number of comparisons for Vol2: ', len(points_real_max_sort))
    return points_real_sort, points_real_max_sort


def min_distance (point, data_points, eps) :
    for p in data_points:
        if np.linalg.norm(point-p)<=eps:
            return 1
    return 0  

def show_eps_pars (data1, data2, value, left, right, step):
    sim=[]
    data_param=[]
    dic_sim={}
    eps=left
    while eps<=right:
        similarity =sum([min_distance(point, data2, eps) for point in data1 ])
        sim.append(similarity)
        data_param.append(eps)
        dic_sim[eps]=similarity
        eps+=step
    print('sim = ',sim)
    print('data_param = ', data_param)
    #print(dic_sim)
    plt.figure(figsize=(8 * 3, 8))
    plt.xlim(0, max(data_param))  # Ось X начинается с 0
    plt.ylim(0, 1.1*value)
    plt.grid(True)
    plt.plot(data_param, sim)
    plt.axhline(y=value, color='red', linestyle='--', linewidth=2)
    plt.xlabel('eps (в пикселях)')
    plt.ylabel('sum_pars')
    plt.show()
    # return min(dic_sim, key=dic_sim.get), min(dic_sim.values())

def save_npy_as_nifty(volume, path_to_save_nii_gz_file):
    volume_tensor = tio.ScalarImage(tensor=torch.tensor(volume).unsqueeze(0))
    volume_tensor.save(path_to_save_nii_gz_file)

def find_and_match_features_SIFT3D(path_to_featExtract,
                                   path_to_nii_gz_file_markup,
                                   path_to_features_file_markup,
                                   path_to_nii_gz_file_test,
                                   path_to_features_file_test,
                                   path_to_output_txt,
                                   num_kp_levels,
                                   sigma_default,
                                   sigma_n_default,
                                   peak_thresh,
                                   max_eigo_thres,
                                   corner_thresh,
                                   matcher_threshold):
    print(subprocess.run([path_to_featExtract,
                          path_to_nii_gz_file_markup,
                          path_to_features_file_markup,
                          path_to_nii_gz_file_test,
                          path_to_features_file_test,
                          path_to_output_txt,
                          str(num_kp_levels),
                          str(sigma_default),
                          str(sigma_n_default),
                          str(peak_thresh),
                          str(max_eigo_thres),
                          str(corner_thresh),
                          str(matcher_threshold)]))

def get_points(path,th):
    out1 = []
    out2 = []
    dist = []
    with open(path, mode='r') as f:
        for line in f:
            i, x, y, z, x1, y1, z1, d = [float(x) for x in line[:-1].split()]
            out1.append([z, y, x])
            out2.append([z1, y1, x1])
            dist.append(d)
    dist = np.array(dist)
    out1 = np.array(out1)
    out2 = np.array(out2)
    return out1[dist<th,:], out2[dist<th,:]
    # return out1, out2

def find_features_SIFT3D(path_to_nii_gz_file, path_to_features_file, path_to_featExtract):
    print(subprocess.run([path_to_featExtract, path_to_nii_gz_file, path_to_features_file]))

def procrustes(src, dst):
    """
    Computes the similarity transform (scale, rotation, translation) that maps src to dst using Procrustes analysis.
    
    Parameters:
        src (np.ndarray): Source points (N, 3).
        dst (np.ndarray): Target points (N, 3).
    
    Returns:
        scale (float): Scaling factor.
        rotation (np.ndarray): 3x3 rotation matrix.
        translation (np.ndarray): 3D translation vector.
    """
    # Compute centroids
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    
    # Center the points
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    
    # Compute scaling factors
    src_scale = np.linalg.norm(src_centered, 'fro')
    dst_scale = np.linalg.norm(dst_centered, 'fro')
    if src_scale == 0:
        scale = 1.0
    else:
        scale = dst_scale / src_scale
    
    # Compute rotation using SVD
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    
    # Ensure proper rotation (handle reflection)
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    
    # Compute translation
    translation = centroid_dst - scale * (rotation @ centroid_src)
    
    return scale, rotation, translation

def ransac_similarity(src, dst, max_iterations=1000, inlier_threshold=0.1, min_samples=3):
    """
    RANSAC algorithm to find the best 3D similarity transform between src and dst points.
    
    Parameters:
        src (np.ndarray): Source points (N, 3).
        dst (np.ndarray): Target points (N, 3).
        max_iterations (int): Maximum number of RANSAC iterations.
        inlier_threshold (float): Distance threshold to consider a point an inlier.
        min_samples (int): Minimum number of points to sample per iteration.
    
    Returns:
        scale (float): Best scaling factor.
        rotation (np.ndarray): Best 3x3 rotation matrix.
        translation (np.ndarray): Best 3D translation vector.
        best_inliers (np.ndarray): Boolean array indicating inliers.
    """
    best_scale = 1.0
    best_rotation = np.eye(3)
    best_translation = np.zeros(3)
    best_inliers = np.zeros(src.shape[0], dtype=bool)
    best_num_inliers = 0
    
    n_points = src.shape[0]
    
    for _ in range(max_iterations):
        # Randomly sample points
        sample_indices = np.random.choice(n_points, min_samples, replace=False)
        src_sample = src[sample_indices]
        dst_sample = dst[sample_indices]
        
        # Check if samples are non-colinear
        v1 = src_sample[1] - src_sample[0]
        v2 = src_sample[2] - src_sample[0]
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue  # Skip colinear samples
        
        # Compute transform using the sample
        try:
            scale, rotation, translation = procrustes(src_sample, dst_sample)
        except:
            continue
        
        # Apply transform to all points
        transformed = scale * (src @ rotation.T) + translation
        
        # Calculate errors
        errors = np.linalg.norm(transformed - dst, axis=1)
        inliers = errors < inlier_threshold
        num_inliers = np.sum(inliers)
        
        # Update best model
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_scale = scale
            best_rotation = rotation
            best_translation = translation
            best_inliers = inliers
            print("impr", best_inliers.sum())
    
    # Refit using all inliers if possible
    if best_num_inliers >= min_samples:
        src_inliers = src[best_inliers]
        dst_inliers = dst[best_inliers]
        best_scale, best_rotation, best_translation = procrustes(src_inliers, dst_inliers)
    
    matrix = np.eye(3,4)
    matrix[:3, :3] = best_rotation*best_scale
    matrix[:3, 3] = best_translation

    return True, matrix, best_inliers

def safe_ransac(points_with_descriptors_test, points_with_descriptors_markup, params):
    # Check if we have enough points
    min_points_required = 4  # Minimum for 3D affine
    
    if len(points_with_descriptors_test) < min_points_required or len(points_with_descriptors_markup) < min_points_required:
        print(f"Warning: Not enough points for affine estimation. Got {len(points_with_descriptors_test)} points, need at least {min_points_required}")
        return False, None, None
    
    try:
        if (params["ransac_type"] == "affine"):
            retval, M, inliers = cv2.estimateAffine3D(np.array(points_with_descriptors_test), np.array(points_with_descriptors_markup), 
                                            ransacThreshold = params["RansacThreshold"], confidence = params["AffineRansacConfidence"])
        elif (params["ransac_type"] == "similarity"):
            retval, M, inliers = ransac_similarity(np.array(points_with_descriptors_markup), np.array(points_with_descriptors_test),
                                            max_iterations=params["SimilarityRansacMaxIterations"], inlier_threshold=params["RansacThreshold"], min_samples=3)
        else:
            print('Wrong ransac version name.')
            sys.exit()
        return retval, M, inliers
        
    except cv2.error as e:
        print(f"OpenCV error in estimateAffine3D: {e}")
        return False, None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False, None, None

def inverse_affine_4x4(matrix_4x4):
    """
    Computes the inverse of a 4x4 affine transformation matrix.
    
    Args:
        matrix_4x4 (np.ndarray): Input 4x4 affine matrix
    
    Returns:
        np.ndarray: 4x4 inverse matrix
    
    Raises:
        ValueError: If matrix is singular or not affine
    """
    # Validate input
    if matrix_4x4.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix")
    if not np.allclose(matrix_4x4[3, :], [0, 0, 0, 1]):
        raise ValueError("Last row must be [0, 0, 0, 1]")
    
    # Extract components
    A = matrix_4x4[:3, :3]
    t = matrix_4x4[:3, 3]
    
    try:
        # Compute inverse of linear transform
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Linear transform component is singular (cannot be inverted)")
    
    # Compute inverse translation
    t_inv = -A_inv @ t
    
    # Build inverse matrix
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = A_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

def read_txt_with_features_SIFT3D(path_to_txt_file_with_, sift_version):
    with open(path_to_txt_file_with_, 'r') as file:
        data_lines = file.readlines()
    if sift_version == "3D_SIFT_CUDA":
        info_string = data_lines[4]
        begin_line_number = 6
        descriptor_size = 81
    elif sift_version == "3DSIFT":
        info_string = data_lines[0]
        begin_line_number = 1
        descriptor_size = 772      
    elif sift_version == "3DSIFT_old_matcher":
        info_string = data_lines[0]
        begin_line_number = 2
        descriptor_size = 772  

    split_info_string = info_string.split(":")
    number_of_points = int(split_info_string[1])
    points_with_descriptors = []
    points_xyz = []
    for i in range(number_of_points - begin_line_number):
        data_line = data_lines[begin_line_number + i].split()
        points_xyz.append([float(data_line[2]), float(data_line[1]), float(data_line[0])])
        points_with_descriptors.append(
            [float(data_line[2]), float(data_line[1]), float(data_line[0]), float(data_line[3])] +
            [float(data_line[i]) for i in range(4, descriptor_size)])
    return (points_xyz, points_with_descriptors, number_of_points)

def run_3DSIFT(markup_volume, transformed_volume, working_folder, metrics_folder_path, params, initial_matrix = np.array([[0]]), init_matrix_given = False):

    path_to_nifty_markup = os.path.join(working_folder, 'markup.nii.gz')
    path_to_nifty_test = os.path.join(working_folder, 'test.nii.gz')

    print("Saving markup gz")
    save_npy_as_nifty(markup_volume, path_to_nifty_markup)
    print("Saving test gz")
    save_npy_as_nifty(transformed_volume, path_to_nifty_test)
    
    if (params["algorithm_points"] == "3DSIFT"):
        path_to_featExtract = params['path_to_featExtract']
        path_to_features_markup_volume = os.path.join(os.path.dirname(path_to_nifty_markup), 'markup_features')
        path_to_features_test_volume = os.path.join(os.path.dirname(path_to_nifty_markup), 'test_features')
        num_kp_levels     = params['num_kp_levels']
        sigma_default     = params['sigma_default']
        sigma_n_default   = params['sigma_n_default']
        peak_thresh       = params['peak_thresh']
        max_eigo_thres    = params['max_eigo_thres']
        corner_thresh     = params['corner_thresh']
        matcher_threshold = params['matcher_threshold']
        find_and_match_features_SIFT3D(path_to_featExtract,
                                       path_to_nifty_markup,
                                       path_to_features_markup_volume,
                                       path_to_nifty_test,
                                       path_to_features_test_volume,
                                       os.path.join(working_folder, 'points.txt'),
                                       num_kp_levels,
                                       sigma_default,
                                       sigma_n_default,
                                       peak_thresh,
                                       max_eigo_thres,
                                       corner_thresh,
                                       matcher_threshold)
        print(f"txt path = {os.path.join(working_folder, 'points.txt')}")
    else:
        print('Wrong sift version name.')
        sys.exit()

    if (params["algorithm_points"] == "3DSIFT"):
        points_with_descriptors_markup, points_with_descriptors_test = get_points(os.path.join(working_folder, 'points.txt'), params['reading_threshold'])
    else:
        markup_points_coordinates, points_with_descriptors_markup, number_of_points_markup = read_txt_with_features_SIFT3D(path_to_features_markup_volume, params["algorithm_points"])
        test_points_coordinates, points_with_descriptors_test, number_of_points_test = read_txt_with_features_SIFT3D(path_to_features_test_volume, params["algorithm_points"])
        del markup_points_coordinates, test_points_coordinates
        points_map = sopostavlenie_create_and_sort_map(np.array(points_with_descriptors_markup), np.array(points_with_descriptors_test))
        sorted_points_map = sort_point_map(points_map)
        points_with_descriptors_markup, points_with_descriptors_test = delete_bad_pairs(sorted_points_map, points_with_descriptors_markup, points_with_descriptors_test)
        del sorted_points_map

    retval, Rt, inliers = safe_ransac(
    points_with_descriptors_test, 
    points_with_descriptors_markup, 
    params
    )

    print(f'retval = {retval}')
    del points_with_descriptors_markup, points_with_descriptors_test

    if retval:
        count = 0
        for inl in inliers:
            if inl ==1:
                count+=1
        print(f'inliers count = {count}')
        Rt = list(Rt)
        Rt.append([0,0,0,1])
        Rt = np.array(Rt)
        Found_matrix = Rt
        if init_matrix_given:
            Rt = Rt@initial_matrix
        inv_Rt = inverse_affine_4x4(Rt)
        print("The found transformation matrix: ")
        print(Rt)
        with open(f'{os.path.join(metrics_folder_path,"matrices.json")}', 'w', encoding='UTF-8') as f:
            json.dump({"matrix":Rt.tolist(),"inv_matrix":inv_Rt.tolist(),"found_matrix":Found_matrix.tolist()}, f)
        print("Number of inliers: ", np.sum(inliers))
        return 1
    else:
        print("Transformation estimation failed")
        return 0

def numpy_parser(num):
    return np.float64(num)

class WrongParam(Exception):
    def __init__(self):
        message = '\nuse_initial_transform_matrix is True, but path_to_initial_transform_matrix_json does not contain suitable json with transformation matrix.\n'
        super().__init__(message)

class InitialTransformError(Exception):
    def __init__(self):
        message = '\nUnable to transform test volume according to given initial transform matrix.\n'
        super().__init__(message)

if __name__ == "__main__":
    markup_volume_path  = sys.argv[1]
    test_volume_path = sys.argv[2]
    processing_folder_path = sys.argv[3]
    metrics_folder_path = sys.argv[4]
    initial_transform_matrix_path = sys.argv[5]
    alg_params_json = sys.argv[6]

    with open(alg_params_json, 'r', encoding='UTF-8') as json_file:
        alg_params = json.load(json_file)

    print(f"Loading markup from path: {markup_volume_path}")
    markup_volume = load_volume_from_dir(markup_volume_path)
    
    if alg_params["use_initial_transform_matrix"]:
        try:
            with open(initial_transform_matrix_path, 'r', encoding='UTF-8') as json_file:
                matrix = np.array(json.load(json_file, parse_float= numpy_parser, parse_int= numpy_parser )["matrix"],dtype=np.float64)
            print(f'Initial matrix reading from given path: \n{matrix}')
            with open(f'{processing_folder_path}/init_transformation.json', 'w', encoding='UTF-8') as f:
                json.dump({"matrix":matrix.tolist()}, f)
        except:
            raise WrongParam()
        python_venv = sys.executable
        print("Creating initialized test volume")
        make_folder(f'{processing_folder_path}/test_initialized/')
        z_shape, y_shape, x_shape = markup_volume.shape
        result = subprocess.run([python_venv, "sample_creator.py",
                                    test_volume_path, f'{processing_folder_path}/init_transformation.json' ,
                                    f'{processing_folder_path}/test_initialized/',
                                    "True",
                                    str(x_shape), str(y_shape), str(z_shape)],
                                    stderr=subprocess.PIPE, text = True)
        print(f'result.stderr = {result.stderr}') 
        if result.stderr:
            print('Stackoverflow while creating test initialized volume')
            raise ImportError("Unable to transform test volume according to given initial transform matrix.")
        print(f"Loading initialized test from path: {processing_folder_path}/test_initialized/")
        test_volume = load_volume_from_dir(f'{processing_folder_path}/test_initialized/')

        print("run_3DSIFT")
        is_good_result = run_3DSIFT(markup_volume, test_volume, processing_folder_path, metrics_folder_path, alg_params, matrix, True)
    else:
        print(f"Loading test from path: {test_volume_path}")
        test_volume = load_volume_from_dir(test_volume_path)

        print("run_3DSIFT")
        is_good_result = run_3DSIFT(markup_volume, test_volume, processing_folder_path, metrics_folder_path, alg_params)

    alg_out_json = f'{processing_folder_path}/alg_out_json.json'
    alg_result = {'output' : is_good_result}

    with open(alg_out_json, "w", encoding="utf-8") as file:
        json.dump(alg_result, file)
    print(f'is_good_result = {is_good_result}')