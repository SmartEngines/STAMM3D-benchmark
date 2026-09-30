# Multimodal 3D image registration benchmark.
Repository for benchmark of rigid multimodal 3D image registration algorithms based on a flexible configurable tool for 3D image registration.

<details><summary>For Arch Linux OS.</summary> 
The registration tool also works on Arch Linux and similar operating systems.
 
**`Important!`** In the case of Arch Linux, you can use these instructions, with the only difference being the command to activate the virtual environment in the terminal (step 2).

</details>

## Configuring the virtual environment for the tool.
To set up the virtual environment, perform the following steps:

0. Clone the repository and find the file `required_libraries.txt` in the `runner` directory of the repository.
 
1.Create a clean virtual environment using [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html):
```shell
python -m venv venv_name
```
 You can also use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) .
 
 2. Activate the environment:
```shell
path\to\venv_name\Scripts\activate.bat
```

<details><summary>For Arch Linux OS.</summary>
 
```bash
path\to\venv_name\bin\activate
```

</details>

3. Run in terminal from the folder `runner` containing `required_libraries.txt` the command:
```shell
pip install -r required_libraries.txt
``` 

## Benchmark dataset acquisition
To run the benchmark, you must download the dataset with 3D multimodal images and a set of transformations, available [here](https://zenodo.org/records/18713582?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImY2Yjg2ZjlmLWUyMjUtNDA0NC1iMzUzLTA2MjJjNTkyMzY5NyIsImRhdGEiOnt9LCJyYW5kb20iOiJhOTgzOGIzOWEwZTg3MDVlOGNhZmU0YWE3N2ZjYTk3MiJ9.NGxRfFNrUElcD_hziCO5C9a9sL_VzLQYfUWjMGRB56Z23J6kv4GMa07nEiPgLzXifJUzMAO7icnFaAFfjvRfAA).
The dataset contents must be placed in a directory of your device’s file system. Example recommended data structure within the repository:
- `data`
  - `benchmark_config.json`
  - `gt_matrix_storage`
    - `package1_transformation.json`
    - `...`
  - `initial_tr_matrix_storage`
    - `package1_init_transformation.json`
    - `...`
  - `markups`
    - `image1`
      - `...`
    - `image2`
      - `...`
    - `...`
- `example_configs`
  - `ready_to_use_configs`
    - `affEnhSIFT`
      - `main_config_affEnhSIFT_benchmark.json`
      - `affEnhSIFT_config.json`
    - `affOrigSITK`
    - `...`
  - `main_config.json`
  - `EnhSIFT_config.json`
  - `...`
- `runner`
  - `main_pipeline.py`
  - `metrics.py`
  - `required_libraries.txt`
  - `sample_creator.py`
  - `SITK.py`
  - `SIFT.py`
  - `...`
- `README.md`

## Interface between benchmark pipeline and registration algorithm.
Main pipeline utilises Command-Line Interface (CLI) to create uniform way of communication with registration algorithm of choice. 
System paths to registration algorithm executable or source code file with interpreter executable and system path to arbitrary registration algorithm configuration file must be provided in main configuration json file. 
Pipeline calls registration algorithm execution with one command string for CLI with combined together system paths in specific order :
```shell
alg_interpreter_path_IF_REQUIRED algorithm_path path_to_fixed_image path_to_moving_image path_to_directory_for_temporary_file_storage path_to_directory_for_registration_result_files_storage path_to_initial_transform_json path_to_algorithm_execution_parameters_file
```
After registration completion algorithm is expected to store two json files: `alg_out.json` with exit code in directory for temporary file storage and `matrices.json` with estimated by algorithm transform matrix in directory for registration result files storage.

<details><summary>Example of `alg_out.json` expected after successful registration algorithm execution.</summary>
 
```json
{"output": 1
}
```
</details>

<details><summary>Example of `matrices.json` expected after successful registration algorithm execution.</summary>

This example shows a 4x4 identity matrix. Your algorithm should output the estimated transformation matrix in the same format.
```json
{"matrix": [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]
}
```
</details>

## Preparing to run the registration algorithm benchmark. 
Before running the benchmark, specify the parameters you need in the main configuration file `main_config.json` provided in the `example_configs` directory of the repository or create a configuration file by its template elsewhere.

<details><summary>Description of main config parameters for example SIFT benchmark `main_config_SIFT_benchmark.json`.</summary>
 
```json
{   "path_to_data_info" : "Path to the directory containing the benchmark dataset (data directory in the repository is preferable).",
    "path_to_data" : "/path/to/repository/data/",

    "path_to_results_info" : "Path to the directory, where a directory with data obtained from the benchmarks’s work will be automatically created.",
    "path_to_results" : "/path/to/results/",

    "experiment_name_info" : ["For each experiment, a directory is created with its name taken from the experiment_name field. During benchmark execution, the current main config and registration algorithm config with timestamp are automatically copied into this directory. All data obtained as a result of benchmark work will also be saved in this directory.",
                              "If a directory with the experiment_name already exists in path_to_results, new runs with the same experiment_name will overwrite its contents."],
    "experiment_name" : "Example_benchmark",

    "algorithm_name_info" : "Defines the name of the directory, NOT THE METHOD itself, which is defined by the registration algorithm executable file.",
    "algorithm_name" : "simEnhSIFT",
    "algorithm_name_examples" : {
        "1" : "SIFT",
        "2" : "SITK",
        "3" : "any_convenient_name_for_the_method"
    },

    "algorithm_execution_parameters_path_info" : ["Path to the registration algorithm config, can be in any format and with any name.",
                                                  "Parameters from this configuration file will be used by registrtion algorithm, make sure they are set properly!"],
    "algorithm_execution_parameters_path" : "/path/to/repository/example_configs/Enhsift_config.json",

    "algorithm_path_info" : "Path to the executable or source code file with the registration algorithm.", 
    "algorithm_path" : "/path/to/repository/runner/SIFT.py",

    "alg_interpreter_path_info" : ["Path to executable of an interpreter for the registration algorithm source code.",
                                   If the registration algorithm is in the form of an executable, alg_interpreter_path must contain an empty string],
    "alg_interpreter_path" : "/path/to/venv_name/Scripts/python.exe",

    "OperationMode_info" : "Do not change for benchmark mode. More details about other tool operation modes are in their descriptions.",
    "OperationMode" : "Benchmark",
    "OperationModeHelp": {
        "mode1" : "Benchmark",
        "mode2" : "ReadyMadeSamples",
        "mode3" : "TwoVolumes"
    },

    "delete_temp_files_info" : "Flag for temp files cleanup after each registration to reduce storage space consumption",
    "delete_temp_files" : true,

    "registered_volumes_writing_info" : ["Flag for saving the 3D images representing results of original 3D images registration.",
                                         "If set to false, the volumes resulting from registration will not be saved as TIFFs; only the parameters of the estimated transform (forward and inverse matrices) will be recorded."],
    "registered_volumes_writing" : false,

    "minimize_padding_info" : ["Flag for choosing the region of interest when saving registration results.",
                               "If false, the resulting volumes after registration will form the bounding rectangular parallelepiped around the registered fixed and moving volumes (logical union of volumes). Added voxels without corresponding intensity values in the original volumes will have zero intensity.",
                               "If true, the resulting volumes show the intersection of the registered fixed and moving volumes with the fixed volume region."],
    "minimize_padding" : true,

    "calculate_metrics_info" : "Flag for calculating metrics. Not used. In Benchmark mode, metrics are always calculated.",
    "calculate_metrics" : true,

    "SelectedMetricsList_info" : "Metrics to calculate. Unneeded ones can be excluded, keeping only desired metrics.",
    "SelectedMetricsList" : ["MSE",
                             "norm_max_corner_TRE",
                             "max_corner_TRE",
                             "norm_BBRE",
                             "BBRE",
                             "norm_geometry_MSE"],

    "Testing_group_list_info" : "Not used in Benchmark mode",
    "Testing_group_list" : [],

    "plot_metrics_info" : "Not used in Benchmark mode",
    "plot_metrics": true,

    "generator_config_path_info" : "Not used in Benchmark mode",
    "generator_config_path" : "",

    "path_to_markup_info" : "Not used in Benchmark mode",
    "path_to_markup" : "",

    "path_to_moving_info" : "Not used in Benchmark mode",
    "path_to_moving" : "",

    "path_to_gt_matrix_json_info" : "Not used in Benchmark mode",
    "path_to_gt_matrix_json" : "",

    "path_to_initial_transform_matrix_json_info" : "Not used in Benchmark mode", 
    "path_to_initial_transform_matrix_json" : "" 
}
```
</details>

**`Important!`** Make sure path provided in `algorithm_execution_parameters_path` leads to proper registration configuration file containing correct registration parameters!

## Running the registration algorithm benchmark. 
From the `runner` directory in console, run the benchmark providing current configuration file:

```shell
python main_pipeline.py /path/to/main_config.json
```

<details><summary>Running the preconfigured example of affine EnhSIFT registration algorithm benchmark. Make sure all paths are set correctly in main and registration algorithm configuration files before the run.</summary>
 
 ```shell
python main_pipeline.py /path/to/repository/example_configs/ready_to_use_configs/affEnhSIFT/main_config_affEnhSIFT_benchmark.json
```
</details>
A copy of the used benchmark config file together with the registration algorithm config file, if specified, will be placed in the selected results directory `path_to_results` to facilitate benchmark reproducibility.

## Registration algorithm benchmarking results.
After benchmark finishes, a CSV text file with benchmark results will be located in the results directory as `path_to_results/stitching_results/benchmark_results.csv` 


## Other operation modes of the registration tool.
Supplementary functions are implemented in two other tool operation modes and grant user ability to perform two independent volumes registrations using an algorithm of choice and to test selected algorithm performance on generated user-specified augmented 3D volumes. Additional information about non-benchmarking functionality will be provided later.
