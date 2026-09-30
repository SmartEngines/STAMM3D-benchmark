import sys
import json
import torch 
import os
import numpy as np
import torchio as tio
import subprocess
import shutil
from data_loader import load_volume_from_dir

# Нахождение ближайших и похожих
def make_folder(folder_path, delete_existing = True):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif delete_existing: 
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def save_npy_as_nifty(volume, path_to_save_nii_gz_file):

    volume_tensor = tio.ScalarImage(tensor=torch.tensor(np.swapaxes(volume, 0, 2)).unsqueeze(0))
    volume_tensor.save(path_to_save_nii_gz_file)

def run_NiftyReg_aladin(path_to_nii_gz_file, path_to_features_file, path_to_featExtract):
    print(subprocess.run([path_to_featExtract, path_to_nii_gz_file, path_to_features_file]))

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

def write_affine_txt(matrix, filename):  
    matrix_list = matrix.tolist()  
    # Open file and write matrix
    with open(filename, 'w') as f:
            for i in range(4):
                row_str = json.dumps(matrix_list[i])
                # Remove brackets and quotes from json output
                row_str = row_str.strip('[]').replace(',', '')
                f.write(row_str)
                
                if i < 3:
                    f.write('\n')

def read_affine_txt_exact(filename, dtype=np.float64):
    matrix = []
    original_strings = []
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Store original strings
            original_strings.append(line)
            
            # Parse values
            values = []
            for val_str in line.split():
                # Parse with full precision
                values.append(float(val_str))
            
            matrix.append(values)
            
            if len(matrix) == 4:
                break
    
    if len(matrix) != 4:
        raise ValueError(f"Expected 4 rows, got {len(matrix)}")
    
    return np.array(matrix, dtype=dtype)

def run_NiftyReg_aladin(markup_volume, transformed_volume, working_folder, metrics_folder_path, params, initial_matrix = np.array([[0]]), init_matrix_given = False):

    path_to_nifty_markup = os.path.join(working_folder, 'markup.nii.gz')
    path_to_nifty_test = os.path.join(working_folder, 'test.nii.gz')
    path_to_al_output = os.path.join(working_folder, 'out.txt')
    path_to_resampled = os.path.join(working_folder, 'outputResult.nii.gz')

    print("Saving markup gz")
    save_npy_as_nifty(markup_volume, path_to_nifty_markup)
    print("Saving test gz")
    save_npy_as_nifty(transformed_volume, path_to_nifty_test)
    
    path_to_aladin = params['path_to_aladin']
    cmd = [path_to_aladin,
           "-ref", path_to_nifty_markup,
           "-flo", path_to_nifty_test,
           "-aff", path_to_al_output,
           "-res", path_to_resampled]
    
    if (params["transform_type"] == "rigid"):
        cmd.append("-rigOnly")
    if init_matrix_given:
        path_to_inaff = os.path.join(working_folder, 'inaff.txt')
        write_affine_txt(initial_matrix, path_to_inaff)
        cmd.extend(["-inaff", path_to_inaff])

    print(subprocess.run(cmd))

    try:
        Rt = read_affine_txt_exact(path_to_al_output, dtype=np.float64)
        inv_Rt = inverse_affine_4x4(Rt)
        print("The found transformation matrix: ")
        print(inv_Rt)
        with open(f'{os.path.join(metrics_folder_path,"matrices.json")}', 'w', encoding='UTF-8') as f:
            json.dump({"matrix":inv_Rt.tolist(),"inv_matrix":Rt.tolist()}, f) #because NiftyReg operates in terms of inverted transforms
        return 1
    except:
        print("Transformation estimation with NiftyReg reg_aladin failed")
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
    print(f"Loading test from path: {test_volume_path}")
    test_volume = load_volume_from_dir(test_volume_path)
    print("run_NiftyReg")

    if alg_params["use_initial_transform_matrix"]:
        try:
            with open(initial_transform_matrix_path, 'r', encoding='UTF-8') as json_file:
                matrix = np.array(json.load(json_file, parse_float= numpy_parser, parse_int= numpy_parser )["matrix"],dtype=np.float64)
            print(f'Initial matrix reading from given path: \n{matrix}')
            inv_matrix = inverse_affine_4x4(matrix)
            is_good_result = run_NiftyReg_aladin(markup_volume, test_volume, processing_folder_path, metrics_folder_path, alg_params, inv_matrix, True)
        except:
            raise WrongParam()
    else:
        is_good_result = run_NiftyReg_aladin(markup_volume, test_volume, processing_folder_path, metrics_folder_path, alg_params)

    alg_out_json = f'{processing_folder_path}/alg_out_json.json'
    alg_result = {'output' : is_good_result}

    with open(alg_out_json, "w", encoding="utf-8") as file:
        json.dump(alg_result, file)
    print(f'is_good_result = {is_good_result}')