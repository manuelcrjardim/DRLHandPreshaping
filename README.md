# Detecting Hand Presahping in a DRL agent
This project analyzes robotic hand trajectories to classify which object is being grasped. It uses data from a simulated environment to train and evaluate machine learning models.

## Environment Setup
#### Prerequisite
This project requires a fully configured and operational [**UniDexGrasp++**](https://github.com/PKU-EPIC/UniDexGrasp2) environment. Please ensure it is installed and working before proceeding.

#### File Configuration
You must replace the following two files in your existing **UniDexGrasp++** directory with the versions provided in this repository:

1. `shadow_hand_grasp.py`
2. `shadow_hand_grasp.yaml`

These files are essential for generating the correct trajectory data needed for the analysis. The `shadow_hand_grasp.py` file is responsible for running the simulation and recording the hand's joint data. The `shadow_hand_grasp.yaml` file contains the specific configurations for the environment, objects, and simulation parameters.

## Execution Order
To run the full pipeline from data processing to visualization, execute the scripts in the following order. It is assumed you have already run the UniDexGrasp++ simulation using the modified files to generate the initial `grasp_data` pickle files.

1. `data_preparation.ipynb`

    - Purpose: This notebook loads the raw trajectory data generated by the simulation, cleans it, and restructures it into a standardized format for analysis.
    - Output: Creates `final_trajectories_randomized.pkl` and `final_successes_randomized.pkl`.

2. `data_analysis.py`

    - Purpose: This is the core analysis script. It trains and evaluates Logistic Regression and MLP models on the prepared data for each timestep of the grasp.
    - Output: Generates CSV files containing detailed model performance metrics, such as `training_and_object_test_accuracies_...csv` and `detailed_instance_predictions.csv`.

3. `render_plots.py`

    - Purpose: Reads the CSV files produced by the analysis script and generates a series of plots visualizing the results, including accuracy curves and confusion matrices.
    - Output: Saves multiple `.png` plot files to the `analysis_plots` directory.

4. `grasp_render.py` (Optional Visualization)

    - Purpose: Creates high-quality, photorealistic renderings of a selected grasp trajectory using MuJoCo. You can configure which object and trajectory to render inside the script.
    - Output: Saves rendered `.png` frames to the `rendered_frames` directory.