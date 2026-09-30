"""
Stitcing metrics.
"""
import json
import numpy as np
import vedo

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

    return [x_max, y_max , z_max]

def distance(point1, point2):
    """
    Returns distances between point1 and point2.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def mse(markup_volume1, volume2, ax = None):
    """
    Mean squared error.
    """
    return np.sum(np.square(markup_volume1 - volume2))/(2 * np.sum(np.square(markup_volume1)))

def geometry_mse(markup_volume, error_transform):
    """
    Mean squared error - transform estimation.
    """
    err_a = error_transform[:3,:3]
    err_b = error_transform[:3,3]

    z, y, x = markup_volume.shape
    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])

    gt_transfrom_points = np.zeros_like(points)
    founded_transfrom_points = np.zeros_like(points)
    distances = []

    print("Corners (gt, founded), Distances:")
    for i, point in enumerate(points):
        gt_transfrom_point = point
        founded_transfrom_point = np.dot(err_a,point) + err_b.T
        gt_transfrom_points[i]= gt_transfrom_point
        founded_transfrom_points[i]= founded_transfrom_point
        distances.append(distance(gt_transfrom_point, founded_transfrom_point))
        print(gt_transfrom_point)
        print(founded_transfrom_point)
        print(distances[i])
        print('---------------------------------')

    # gt_transfrom_points_vedo = vedo.Points(gt_transfrom_points, r=5)
    # founded_transfrom_points_vedo = vedo.Points(founded_transfrom_points, r=5)
    # plotter = vedo.Plotter()
    # plotter.add(gt_transfrom_points_vedo.color('red'))
    # plotter.add(founded_transfrom_points_vedo.color('blue'))
    # #plotter.show(f'{float((np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2), float(((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5), float(max(distances))}', axes = 1)
    # plotter.remove(gt_transfrom_points_vedo.color('red'),
    #                founded_transfrom_points_vedo.color('blue'))
    
    return (np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2, ((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5, ((np.square(distances)).mean(axis = None))**0.5, max(distances)/np.linalg.norm([z, y, x]) , max(distances)

def geometry_mse_calibration(markup_volume, gt_transform):
    """
    Mean squared error - transform estimation.
    """
    gt_a = gt_transform[:3,:3]
    gt_b = gt_transform[:3,3]
    print(gt_a)
    print(gt_b)
    z, y, x = markup_volume.shape
    points = np.array([[0, 0, 0],[0, 0, z],
                        [0, y, 0],[0, y, z],
                        [x, 0, 0],[x, 0, z],
                        [x, y, 0],[x, y, z]])
    
    gt_transfrom_points = np.zeros_like(points)

    distances = []

    print("Corners |gt - founded| = Distances:")
    for i, point in enumerate(points):
        gt_transfrom_point = np.dot(gt_a,point) + gt_b.T
        print(np.dot(gt_a,point) + gt_b.T)
        gt_transfrom_points[i]= gt_transfrom_point
        distances.append(distance(gt_transfrom_point, point))
        print(f'|{point} - {gt_transfrom_point}| = {distances[i]}')
        print('---------------------------------')

    return (np.square(distances)).mean(axis = None) / np.linalg.norm([z, y, x])**2, ((np.square(distances)).mean(axis = None) / (np.linalg.norm([z, y, x])**2))**0.5, max(distances)

def final_rotation_angle(gt_transform, founded_transform):
    """
    Returns final rotation angle between 
    the column vectors of gt transformation matrix
    and founded transformation matrix. ???
    """
def pad_to_shape(array, t_shape):
    """
    Returns the expanded volume to the specified target shape.
    """
    padding = [(0, t - s) for s,t in zip(array.shape, t_shape)]
    return np.pad(array, padding, mode = 'constant', constant_values = 0)

def show_volumes(v1, v2):
    plotter = vedo.Plotter()
    vedo_v1 = vedo.Volume(v1)
    vedo_v1.permute_axes(2,1,0)
    plotter.add(vedo_v1)
    vedo_v2 = vedo.Volume(v2)
    vedo_v2.permute_axes(2,1,0)
    plotter.add(vedo_v2)
    plotter.show(vedo_v1.alpha([0,0, 0.05, 0]).color('red'), vedo_v2.alpha([0,0, 0.05, 0]).color('blue'), axes = 1)
    plotter.remove(vedo_v1, vedo_v2)

def apply_transform(volume, matrix):
    linear_transform = vedo.LinearTransform()
    linear_transform.matrix = matrix
    #print(f'transform matrix = {matrix}')
    new_volume_shape = np.array(get_bndbox(volume.shape, matrix))
    #print(f'new volume shape (x, y, z) = {new_volume_shape}')
    
    vedo_volume = vedo.Volume(volume)
    vedo_volume.permute_axes(2,1,0)
    try:
        vedo_volume.apply_transform(linear_transform, fit = True, interpolation = 'linear')
    except:
        pass

    empty_arr = np.zeros(new_volume_shape)
    vedo_volume_new = vedo.Volume(empty_arr, origin = [0,0,0], dims = empty_arr.shape)
    vedo_volume_new.resample_data_from(vedo_volume)
    vedo_volume = vedo_volume_new
    vedo_volume.permute_axes(2,1,0)
    return vedo_volume.tonumpy().astype(np.float32)

def calculate_metrics(sample_result_folder_path,
                      markup_volume, transformed_volume,
                      error_transform, metric_volume, metric_name_list, registration_fail = False, volume_creation_fail = False ):
    """
    Calculates and writes metrics in json: intensity mse, geometry mse, geometry rmse, maximum deviation.
    """
    metrics_store = {}
    a = np.array(["norm_geometry_MSE","norm_BBRE","BBRE","max_corner_TRE","norm_max_corner_TRE"])
    b = np.array(metric_name_list)
    if registration_fail:
        metrics_store["MSE"] =  -1
        metrics_store["norm_geometry_MSE"] =  -1
        metrics_store["norm_BBRE"] =  -1
        metrics_store["BBRE"] =  - 1
        metrics_store["max_corner_TRE"] = - 1
        metrics_store["norm_max_corner_TRE"] = -1
        with open(f'{sample_result_folder_path}metrics.json', 'w', encoding='UTF-8') as f:
            json.dump(metrics_store, f)
        return 0
    
    if np.isin(a, b).any():
        calculated_norm_geometry_mse, norm_BBRE, bbre, norm_max_dev, max_dev = geometry_mse(metric_volume, error_transform)
        print(f'normalized_geometry_MSE = {calculated_norm_geometry_mse}')
        print(f'normalized_BBRE = {norm_BBRE}')
        print(f'BBRE = {bbre}')
        print(f'normalized_max_corner_TRE = {norm_max_dev}')
        print(f'max_corner_TRE = {max_dev}')
        metrics_store["norm_geometry_MSE"] = float(calculated_norm_geometry_mse)
        metrics_store["norm_BBRE"] = float(norm_BBRE)
        metrics_store["BBRE"] = float(bbre)
        metrics_store["norm_max_corner_TRE"] = float(norm_max_dev)
        metrics_store["max_corner_TRE"] = float(max_dev)

    if volume_creation_fail:
        metrics_store["MSE"] =  -1
        with open(f'{sample_result_folder_path}metrics.json', 'w', encoding='UTF-8') as f:
            json.dump(metrics_store, f)
        return 0
    
    if "MSE" in metric_name_list:
        try:
            calculated_mse = mse(markup_volume, transformed_volume)
        except:
            calculated_mse = -1
        print(f'MSE = {calculated_mse}')
        metrics_store["MSE"] = float(calculated_mse)

    with open(f'{sample_result_folder_path}metrics.json', 'w', encoding='UTF-8') as f:
        json.dump(metrics_store, f)
