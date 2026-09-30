"""
Main pipeline of testing stitching volumes methods.
"""
import json
import sys
import os
import shutil
import numpy as np
from cv2 import imwrite
import metrics
import subprocess
from timer import timer
import tracemalloc
import psutil
import time
from joblib import Parallel, delayed
from datetime import datetime
from data_loader import load_volume_from_dir, get_volume_shape
from results_visualization import ReadyMadeSamplesParser,BenchmarkMetricParser

def make_folder(folder_path, delete_existing = True):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    elif delete_existing: 
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)

def timestamped_config_safe_copy(source, alg,  alg_source, target_dir):
    base_name = os.path.basename(source)
    alg_base_name = os.path.basename(alg_source)
    target_path = os.path.join(target_dir,f'{alg}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_configs')
    if os.path.exists(target_path):
        counter = 1
        while True:
            new_path = target_path + f'{counter:03d}'
            if not os.path.exists(new_path):
                break
            counter += 1
        target_path = new_path
    make_folder(target_path)
    shutil.copy(source, os.path.join(target_path,base_name))
    shutil.copy(alg_source, os.path.join(target_path,alg_base_name))
    pass

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

def crop_and_pad_to_shape(array, t_shape):
    """
    Returns croped and expanded volume, matching the specified target shape.
    """
    slices = tuple(slice(0, min(a, t)) for a, t in zip(array.shape, t_shape))
    cropped = array[slices]
    padding = [(0, max(0, t - s)) for s, t in zip(cropped.shape, t_shape)]
    return np.pad(cropped, padding, mode='constant', constant_values = 0)

def calculate_metrics_transform_fail(f_markup_path,f_sample_result_folder_path,f_j,f_SelectedMetricsList):
    f_metric_volume = load_volume_from_dir(f_markup_path)
    metrics.calculate_metrics(f'{f_sample_result_folder_path}/metrics/{f_j}/',
                                0, 0,
                                0, f_metric_volume, f_SelectedMetricsList, registration_fail=True)
    del f_metric_volume

def calculate_metrics_volume_creation_fail(f_markup_path,f_sample_result_folder_path,f_j,f_deviation_matrix,f_SelectedMetricsList):
    f_metric_volume = load_volume_from_dir(f_markup_path)
    metrics.calculate_metrics(f'{f_sample_result_folder_path}/metrics/{f_j}/',
                                0, 0,
                                f_deviation_matrix, f_metric_volume, f_SelectedMetricsList, volume_creation_fail= True)
    del f_metric_volume
    

if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path, 'r', encoding='UTF-8') as json_file:
        config = json.load(json_file)
    python_venv = sys.executable
    alg_interpreter_path = config["alg_interpreter_path"]
    data_folder = config["path_to_data"]
    main_folder = config["path_to_results"]
    OperationMode = config["OperationMode"]
    exp_path = os.path.join(main_folder,f'{config["experiment_name"]}/')
    if not os.path.exists(exp_path):
        if OperationMode != "ReadyMadeSamples":
            make_folder(exp_path)
        else:
            print('OperationMode == "ReadyMadeSamples", but premade folder with given in main_config.json "experiment_name" does not exist in "path_to_results"')
            print('Make sure you have given correct path and folder name to premade transformed volumes')
            sys.exit(0)
    transformed_volumes_path = os.path.join(main_folder, f'{config["experiment_name"]}', "transformed_volumes/")
    Testing_group_index_list = config["Testing_group_list"]
    minimize_padding = config["minimize_padding"]
    algorithm = config["algorithm_name"]
    algorithm_config_path = config["algorithm_execution_parameters_path"]
    algorithm_path = config["algorithm_path"]
    registered_volumes_writing = config["registered_volumes_writing"]
    calculate_metrics = config["calculate_metrics"]
    generator_config_path = config["generator_config_path"]
    timestamped_config_safe_copy(config_path, algorithm, algorithm_config_path,  exp_path)
    lists_of_paths_to_test_volumes = []
    markup_path_list = []
    markup_package_list = []
    path_to_gt_matrix_list = []
    path_to_init_transform_matrix_list = []
    if OperationMode == "Benchmark":
        calculate_metrics = True
        benchmark_config_path = os.path.join(data_folder,"benchmark_config.json")
        with open(benchmark_config_path, 'r', encoding='UTF-8') as json_file:
            benchmark_config = json.load(json_file)
        for package in benchmark_config :
            if benchmark_config[package]["skip_in_benchmark"]:
                continue 
            markup_package_list.append(package)
            markup_path_list.append(os.path.join(data_folder, benchmark_config[package]["markup_path"]))
            lists_of_paths_to_test_volumes.append([os.path.join(data_folder, benchmark_config[package]["test_path"])])
            path_to_gt_matrix_list.append(os.path.join(data_folder,benchmark_config[package]["gt_matrix_path"]))
            path_to_init_transform_matrix_list.append(os.path.join(data_folder,benchmark_config[package]["initial_tr_matrix_path"]))
    elif OperationMode == "ReadyMadeSamples":
        markup_path_list.append("blanc")
        markup_package_list.append("blanc")
        path_to_init_transform_matrix_list.append("blanc")
        test_volumes_path_list = []
        Testing_group_list = []    
        if len(Testing_group_index_list)==0:
            for name in os.listdir(transformed_volumes_path):
                if (name[0:2]=='gr'):
                    Testing_group_list.append(name)
        else:
            for name in os.listdir(transformed_volumes_path):
                if (int(name[2]) in Testing_group_index_list):
                    Testing_group_list.append(name)
        Testing_group_list = sorted(Testing_group_list, key = lambda x: int(str(x.rsplit("_")[0])[2:]))
        
        for gr_name in Testing_group_list:
            group_path = os.path.join(transformed_volumes_path,gr_name)
            if int(gr_name[-1])==2:
                for j in sorted(os.listdir(group_path),key = lambda x: int(x.rsplit("_",1)[1])):
                    for k in sorted(os.listdir(os.path.join(group_path,j)),key = lambda x: int(x.rsplit("_",1)[1])):
                        test_volumes_path_list.append(os.path.join(group_path,j,k+"/"))
            elif int(gr_name[-1])==1:
                for j in sorted(os.listdir(group_path),key = lambda x: int(x.rsplit("_",1)[1])):
                    test_volumes_path_list.append(os.path.join(group_path,j+"/"))
            elif int(gr_name[-1])==0:
                test_volumes_path_list.append(group_path+"/")
        lists_of_paths_to_test_volumes.append(test_volumes_path_list)
    elif OperationMode == "TwoVolumes":
        markup_path_list.append(config["path_to_markup"])
        path_to_moving = config["path_to_moving"]
        markup_package_list.append("custom_markup")
        lists_of_paths_to_test_volumes.append([path_to_moving])
        path_to_init_transform_matrix_list.append(config["path_to_initial_transform_matrix_json"])
        if calculate_metrics:
            path_to_gt_matrix_list.append(config["path_to_gt_matrix_json"])
    
    print(f'Selected algorithm is {algorithm}.')
    processing_folder_path = os.path.join(exp_path, 'processing/')
    make_folder(processing_folder_path,delete_existing=False)
    alg_processing_folder_path = os.path.join(exp_path, 'processing', algorithm + '/')
    make_folder(alg_processing_folder_path)
    stitching_results_folder_path = os.path.join(exp_path, 'stitching_results/')
    make_folder(stitching_results_folder_path, delete_existing=False) 
    alg_stitching_results_folder_path = os.path.join(exp_path, 'stitching_results', algorithm + '/')
    make_folder(alg_stitching_results_folder_path) 

    for index,(markup_package,current_markup_path, paths_to_test_volumes) in enumerate(zip(markup_package_list, markup_path_list,lists_of_paths_to_test_volumes)):
        if OperationMode != "ReadyMadeSamples":
            package_processing_folder_path = os.path.join(exp_path, 'processing', algorithm,  markup_package + '/')
            make_folder(package_processing_folder_path)
            package_stitching_results_folder_path = os.path.join(exp_path, 'stitching_results', algorithm, markup_package + '/')
            make_folder(package_stitching_results_folder_path)
        number_of_samples = len(paths_to_test_volumes)
        for i in range(number_of_samples):
            print("--"*40)
            if OperationMode == "ReadyMadeSamples":
                sample_processing_folder_path = os.path.join(alg_processing_folder_path,paths_to_test_volumes[i].split("transformed_volumes",1)[-1].lstrip("/"))
            else:
                sample_processing_folder_path = package_processing_folder_path
            print('sample_processing_folder_path = ', sample_processing_folder_path)
            os.makedirs(sample_processing_folder_path, exist_ok=True)
            if OperationMode == "ReadyMadeSamples":
                sample_result_folder_path = os.path.join(alg_stitching_results_folder_path,paths_to_test_volumes[i].split("transformed_volumes",1)[-1].lstrip("/"))
            else:
                sample_result_folder_path = package_stitching_results_folder_path
            print('sample_result_folder_path = ', sample_result_folder_path)
            os.makedirs(sample_result_folder_path, exist_ok=True)

            if OperationMode == "ReadyMadeSamples":
                with open(paths_to_test_volumes[i]+"/tr_gen.json", 'r', encoding='UTF-8') as json_file:
                    sample_config = json.load(json_file)
                testing_package = sample_config["testing_package"]
                markup_path = os.path.join(data_folder,"markups",testing_package+"/")
                repeat = sample_config["repeat"]
                if calculate_metrics :
                    markup_offset = sample_config["markup_offset"]
                    given_affine_matrix = np.array(sample_config["target_transformation"])
                    inv_gt_matrix = np.array(sample_config["transformation"])
            else:
                markup_path = current_markup_path
                repeat = 1
                if calculate_metrics:
                    with open(path_to_gt_matrix_list[index], 'r', encoding='UTF-8') as json_file:
                        gt_json = json.load(json_file)
                    if "inv_gt_matrix" in gt_json:
                        inv_gt_matrix = np.array(gt_json["inv_gt_matrix"])
                    elif "gt_matrix" in gt_json:
                        inv_gt_matrix = inverse_affine_4x4(np.array(gt_json["gt_matrix"]))
                    else:
                        metric_names = ["norm_geometry_MSE",
                                        "norm_BBRE",
                                        "BBRE",
                                        "norm_max_corner_TRE",
                                        "max_corner_TRE"]
                        for metric in config["SelectedMetricsList"]:
                            if metric in metric_names:
                                print("Unable to extract inv_gt_matrix or gt_matrix out of provided gt matrix json!")
                                sys.exit()
                        inv_gt_matrix = np.eye(4)
                        

            make_folder(f'{sample_result_folder_path}/metrics/') 

            for j in range(repeat):
                os.mkdir(f'{sample_result_folder_path}/metrics/{j}/')
                single_repeat_metrics_folder_path = f'{sample_result_folder_path}/metrics/{j}/'
                os.mkdir(f'{sample_processing_folder_path}/{j}/')
                single_repeat_processing_folder_path = f'{sample_processing_folder_path}/{j}/'
                info = {
                    "alg_path" : algorithm_path,
                    "markup_path" : markup_path,
                    "test_volume_path" : paths_to_test_volumes[i],
                    "processing_folder_path" : single_repeat_processing_folder_path,
                    "initial_tr_matrix_path" : path_to_init_transform_matrix_list[index],
                    "alg_config_path" : algorithm_config_path
                }
                tracemalloc.start()
                with timer(unit = "ms") as t:
                    if alg_interpreter_path == "":
                        subpr_arg_seqence = [algorithm_path,markup_path, paths_to_test_volumes[i],single_repeat_processing_folder_path, single_repeat_metrics_folder_path, path_to_init_transform_matrix_list[index], algorithm_config_path]
                    else:
                        subpr_arg_seqence = [alg_interpreter_path, algorithm_path,markup_path, paths_to_test_volumes[i], single_repeat_processing_folder_path, single_repeat_metrics_folder_path, path_to_init_transform_matrix_list[index], algorithm_config_path]
                        info.update({"alg_interpreter_path" : alg_interpreter_path})
                    proc = subprocess.Popen(subpr_arg_seqence, stderr=subprocess.PIPE, text=True)
                    p=psutil.Process(proc.pid)
                    children = p.children(recursive=True)
                    if children:
                        #print(f" number of children = {len(children)},\n {children}")
                        p = children[0]
                    peak_mem = 0
                    while proc.poll() is None:
                        try:
                            mem = p.memory_info().rss/(10**6)
                            peak_mem = max(peak_mem, mem)
                            time.sleep(0.01)
                        except psutil.NoSuchProcess:
                            break
                    stdout, stderr = proc.communicate()
                alg_time = t.elapse/1000
                current, peak = tracemalloc.get_traced_memory()
                print(f"{algorithm} timer = {alg_time:.6f} seconds")
                print(f"peak {algorithm} memory = {peak_mem:.4f} MB")
                time_mem_params = {"Time sec":alg_time,"Overall_peak_mem_MB":peak_mem,"Peak_memory_of_pipeline_MB":peak/10**6,"markup":markup_path,"test":paths_to_test_volumes[i]}
                time_mem_params.update(info)
                try:
                    with open(algorithm_config_path, 'r', encoding='UTF-8') as json_file:
                        alg_params = json.load(json_file)
                    time_mem_params.update(alg_params)
                except:
                    print(f"Unable to read {algorithm} json configuration file from:\n{algorithm_config_path}")
                    pass
                time_json = f'{sample_result_folder_path}/metrics/{j}/time_mem.json'
                with open(time_json, "w", encoding="utf-8") as file:
                    json.dump(time_mem_params, file)
                print(f'result_stdout = {stdout}')
                print(f'result_stderr = {stderr}')

                try:
                    alg_out_json = f'{single_repeat_processing_folder_path}/alg_out_json.json'
                    with open(alg_out_json, 'r', encoding='UTF-8') as json_file:
                        alg_out = json.load(json_file)
                    print(f'alg_out = {alg_out}')
                    if alg_out["output"] == 0:
                        print(f'Failed to estimate transform with {algorithm}')
                        if calculate_metrics : calculate_metrics_transform_fail(markup_path,sample_result_folder_path,j,config["SelectedMetricsList"])
                        continue
                except:
                    if calculate_metrics : calculate_metrics_transform_fail(markup_path,sample_result_folder_path,j,config["SelectedMetricsList"])
                    print(f'Stackoverflow while calculating matrix with {algorithm}')
                    continue

                with open(f'{os.path.join(single_repeat_metrics_folder_path,"matrices.json")}', 'r', encoding='UTF-8') as json_file:
                    transformation_estimated = json.load(json_file)["matrix"]
                transformation_estimated = np.array(transformation_estimated)
                np.set_printoptions(precision = 8, suppress=True)
                if calculate_metrics :
                    if OperationMode == "ReadyMadeSamples":
                        print("Augmentation matrix")
                        print(given_affine_matrix)
                    print("GT matrix")
                    print(inverse_affine_4x4(inv_gt_matrix))
                    print("Deviation matrix")
                    deviation_matrix = transformation_estimated @ inv_gt_matrix
                    print(deviation_matrix)
                print(f"Estimated by {algorithm} matrix")
                print(transformation_estimated)
                x_shape, y_shape, z_shape = get_volume_shape(markup_path)[::-1]
                if registered_volumes_writing or (calculate_metrics and "MSE" in config["SelectedMetricsList"]):
                    make_folder(f'{sample_result_folder_path}/test_transformed_{j}/')
                    result = subprocess.run([python_venv, "sample_creator.py",
                                                paths_to_test_volumes[i], f'{single_repeat_metrics_folder_path}/matrices.json' ,
                                                f'{sample_result_folder_path}/test_transformed_{j}/',
                                                str(minimize_padding),
                                                str(x_shape), str(y_shape), str(z_shape)],
                                                stderr=subprocess.PIPE, text = True)
                    print(f'result.stderr = {result.stderr}') 
                    if result.stderr:
                        if calculate_metrics : calculate_metrics_volume_creation_fail(markup_path,sample_result_folder_path,j,deviation_matrix,config["SelectedMetricsList"])
                        print('Stackoverflow while creating markup transformed volume')
                        continue
                    try:
                        test_transformed_volume = load_volume_from_dir(f'{sample_result_folder_path}/test_transformed_{j}')
                        if minimize_padding == True:
                            padded_test = test_transformed_volume.astype(np.float32)
                        else:
                            with open(f'{os.path.join(single_repeat_metrics_folder_path,"origin_negative_offset.json")}', 'r', encoding='UTF-8') as json_file:
                                origin_offset = json.load(json_file)["origin_offset"]
                            markup_shape = get_volume_shape(markup_path)
                            target_shape = np.maximum(get_volume_shape(f'{sample_result_folder_path}/test_transformed_{j}'), np.array([markup_shape[0]+origin_offset[2],markup_shape[1]+origin_offset[1],markup_shape[2]+origin_offset[0]]))
                            padded_test = crop_and_pad_to_shape(test_transformed_volume, target_shape).astype(np.float32)
                        del test_transformed_volume
                        print('Saving padded_test...')
                        make_folder(f'{sample_result_folder_path}/padded_test_{j}/')
                        for k, Slice in enumerate(padded_test):
                            save_path = os.path.join(f'{sample_result_folder_path}/padded_test_{j}/{k:04d}.tif')
                            imwrite(save_path, Slice) # from cv2
                    except:
                        if calculate_metrics : calculate_metrics_volume_creation_fail(markup_path,sample_result_folder_path,j,deviation_matrix,config["SelectedMetricsList"])
                        print('Stackoverflow while creating padded test volume')
                        continue

                    try:
                        markup_volume = load_volume_from_dir(markup_path)
                        if minimize_padding == True:
                            padded_markup = markup_volume.astype(np.float32)
                        else:
                            padded_markup = np.zeros(list(target_shape)).astype(np.float32)
                            padded_markup[origin_offset[2]:origin_offset[2]+markup_volume.shape[0],origin_offset[1]:origin_offset[1]+markup_volume.shape[1],origin_offset[0]:origin_offset[0]+markup_volume.shape[2]] = markup_volume
                        del markup_volume
                        print('Saving padded_markup...')
                        make_folder(f'{sample_result_folder_path}/padded_markup_{j}/')
                        for k, Slice in enumerate(padded_markup):
                            save_path = os.path.join(f'{sample_result_folder_path}/padded_markup_{j}/{k:04d}.tif')
                            imwrite(save_path, Slice) # from cv2 
                    except:
                        if calculate_metrics : calculate_metrics_volume_creation_fail(markup_path,sample_result_folder_path,j,deviation_matrix,config["SelectedMetricsList"])
                        print('Stackoverflow while creating padded markup volume')
                        continue

                    try:
                        if calculate_metrics :
                            metric_volume = load_volume_from_dir(markup_path)
                            metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                                    padded_markup, padded_test,
                                                    deviation_matrix, metric_volume, config["SelectedMetricsList"])
                            del metric_volume
                    except Exception as e:
                            print(f'Error while calculating metrics:\n{e}')
                    if not registered_volumes_writing:
                        shutil.rmtree(f'{sample_result_folder_path}/test_transformed_{j}/')
                        shutil.rmtree(f'{sample_result_folder_path}/padded_markup_{j}/')
                        shutil.rmtree(f'{sample_result_folder_path}/padded_test_{j}/')
                    else:
                        try:
                            print('Saving join...')
                            # make_folder(f'{sample_result_folder_path}/join_{j}/')
                            # join = padded_markup+padded_test
                            # for k, Slice in enumerate(join):
                            #     save_path = os.path.join(f'{sample_result_folder_path}/join_{j}/{k:04d}.tif')
                            #     imwrite(save_path, Slice) # from cv2 

                            np.clip(padded_markup, a_min=0, a_max=None, out=padded_markup)
                            np.clip(padded_test, a_min=0, a_max=None, out=padded_test)

                            # max_value = percentile_across_arrays([volume1,volume2], 99)
                            # print('99-percentile = ', max_value)
                            max_value1 = np.percentile(padded_markup, 99)
                            max_value2 = np.percentile(padded_test, 99)
                            print('99-percentile1 = ', max_value1, '99-percentile2 = ', max_value2 )

                            uint_volume1 =  np.floor(255 * padded_markup /max_value1)
                            del padded_markup
                            uint_volume2 = np.floor( 255 * padded_test/max_value2)
                            del padded_test

                            np.clip(uint_volume1, a_min=0, a_max=255, out=uint_volume1)
                            np.clip(uint_volume2, a_min=0, a_max=255, out=uint_volume2)

                            def make_color_slice(slice_number, uint_volume1, uint_volume2, save_path):
                                color_slice = np.stack([uint_volume2[slice_number],uint_volume2[slice_number],uint_volume1[slice_number]],axis=-1).astype(np.uint8)
                                save_path_slice = os.path.join(f'{save_path}/{slice_number:04d}.tif')
                                imwrite(save_path_slice, color_slice)
                            save_path = os.path.join(f'{sample_result_folder_path}', f'colored_joining_volume_{j}')
                            make_folder(save_path)
                            Parallel(n_jobs=-2, prefer="threads")(delayed(make_color_slice)(slice_number, uint_volume1, uint_volume2, save_path) for slice_number in range(uint_volume2.shape[0]))
                        except:
                            print('Stackoverflow while creating joined volume')
                elif calculate_metrics:
                    try:
                        metric_volume = load_volume_from_dir(markup_path)
                        metrics.calculate_metrics(f'{sample_result_folder_path}/metrics/{j}/',
                                                0, 0,
                                                deviation_matrix, metric_volume, config["SelectedMetricsList"])
                        del metric_volume
                    except Exception as e:
                            print(f'Error while calculating metrics:\n{e}')
                if config["delete_temp_files"]:
                    shutil.rmtree(single_repeat_processing_folder_path)

    if calculate_metrics:
        if OperationMode == "Benchmark":
            BenchmarkMetricParser(stitching_results_folder_path, config["SelectedMetricsList"],error_threshold = 0.144, n_packages = 19)
        elif OperationMode == "ReadyMadeSamples":
            ReadyMadeSamplesParser(config_path, generator_config_path)
