# IMPORTS

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

################################ LOAD FINAL DATASETS ################################

with open('final_trajectories_randomized.pkl', 'rb') as f:
    final_trajectories = pickle.load(f)

with open('final_successes_randomized.pkl', 'rb') as f:
    final_successes = pickle.load(f)

################################ DATA PREPARATION ################################

def get_object_type(full_object_name):
    parts = full_object_name.split('/')
    type_part = parts[1].split('-')
    if type_part:
        return type_part[0]

# Initialize dictionaries to store accuracies for plotting
lr_accuracies_by_type_and_timestep = defaultdict(dict)
mlp_accuracies_by_type_and_timestep = defaultdict(dict) 

# Configuration
max_timesteps = 200
num_joints = 22

# Initialize list to store data for CSV
all_csv_data = []
all_full_object_names_global_csv = sorted(list(final_trajectories.keys()))

lr_individual_predictions = []
mlp_individual_predictions = []

search_freq = 50


# Prepare CSV
for target_timestep in range(1, max_timesteps + 1):    
    # Initialize CSV row for the current timestep with NaNs
    current_csv_row = {'Timestep': target_timestep}
    current_csv_row['LR_Train_Accuracy'] = np.nan
    current_csv_row['MLP_Train_Accuracy'] = np.nan
    current_csv_row['LR_Overall_Test_Accuracy'] = np.nan
    current_csv_row['MLP_Overall_Test_Accuracy'] = np.nan
    for name in all_full_object_names_global_csv:
        clean_name = name.replace('/', '_').replace('-', '_')
        current_csv_row[f'LR_Test_Accuracy_Object_{clean_name}'] = np.nan
        current_csv_row[f'MLP_Test_Accuracy_Object_{clean_name}'] = np.nan
    all_csv_data.append(current_csv_row)

################################ LOGISTIC REGRESSION HYPERPARAMETER SEARCH ################################

counter = 0

param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100], 
    'solver': ['liblinear', 'saga'], 
    'penalty': ['l1', 'l2'] 
}

base_lr_for_grid = LogisticRegression(max_iter=5000, random_state=42, multi_class='auto')

best_lr_params_list = [] 
most_common_params_dict = {} 

for target_timestep in range(1, max_timesteps+1, search_freq):
    print(f"\nProcessing timestep: {target_timestep}")

    features_for_timestep = []
    labels_for_timestep_full_names = []
    for object_name_full, trajectories_list in final_trajectories.items():
        for trajectory_dict in trajectories_list:
            joint_values = trajectory_dict[target_timestep]           
            features_for_timestep.append(joint_values)
            labels_for_timestep_full_names.append(object_name_full)
    
    X_timestep = np.array(features_for_timestep)
    label_encoder_full_names = LabelEncoder()
    y_encoded_full_names = label_encoder_full_names.fit_transform(labels_for_timestep_full_names)
    num_distinct_full_classes = len(label_encoder_full_names.classes_)

    label_counts_full_names = Counter(y_encoded_full_names)

    X_train, X_test, y_train_full_encoded, y_test_full_encoded, _, test_set_full_names_original = train_test_split(
        X_timestep, y_encoded_full_names, labels_for_timestep_full_names,
        test_size=0.2, random_state=42, stratify=y_encoded_full_names
        )
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    actual_types_test = []
    unique_actual_types_in_test = []
    actual_types_test = [get_object_type(name) for name in test_set_full_names_original]
    unique_actual_types_in_test = sorted(list(set(actual_types_test)))

    y_pred_lr_full_encoded = [] 

    # Perform GridSearchCV to find the best Logistic Regression model
    grid_search_lr = GridSearchCV(estimator=base_lr_for_grid,
                                  param_grid=param_grid_lr,
                                  cv=3, 
                                  scoring='accuracy',
                                  n_jobs=-1,
                                  verbose=0) 
    
    grid_search_lr.fit(X_train_scaled, y_train_full_encoded)

    best_lr_params_list.append(grid_search_lr.best_params_)
    print(f"Timestep {target_timestep}: Best LR Params: {grid_search_lr.best_params_}")
    print(f"Timestep {target_timestep}: Best LR CV Score on Train Data: {grid_search_lr.best_score_:.4f}")

    model_lr = grid_search_lr.best_estimator_ 
    
    y_pred_lr_train_encoded = model_lr.predict(X_train_scaled) 
    all_csv_data[counter]['LR_Train_Accuracy'] = accuracy_score(y_train_full_encoded, y_pred_lr_train_encoded)
    print(f"Timestep {target_timestep}: Logistic Regression Train Accuracy (Best Model) = {all_csv_data[counter]['LR_Train_Accuracy']:.4f}")

    y_pred_lr_full_encoded = model_lr.predict(X_test_scaled) 
    all_csv_data[counter]['LR_Overall_Test_Accuracy'] = accuracy_score(y_test_full_encoded, y_pred_lr_full_encoded)
    print(f"Timestep {target_timestep}: Logistic Regression Overall Test Accuracy (Best Model) = {all_csv_data[counter]['LR_Overall_Test_Accuracy']:.4f}")

    counter += 1

# Determine the most frequent best hyperparameters
# Convert dicts to a hashable type (tuple of sorted items) for Counter
param_tuples = [tuple(sorted(params.items())) for params in best_lr_params_list]
param_counts = Counter(param_tuples)
most_common_params_tuple = param_counts.most_common(1)[0][0]
most_common_params_dict = dict(most_common_params_tuple)
print("\n--- Most Frequent Best Logistic Regression Hyperparameters Overall ---")
print(f"Parameters: {most_common_params_dict}")
print(f"Frequency: {param_counts.most_common(1)[0][1]} out of {len(best_lr_params_list)} timesteps processed.")



################################################ TRAIN LOGISTIC REGRESSION ################################################
counter = 0

for target_timestep in range(1, max_timesteps + 1):
    print(f"\nProcessing timestep: {target_timestep}")

    features_for_timestep = []
    labels_for_timestep_full_names = []
    instance_ids_for_timestep = [] 

  
    for object_name_full, trajectories_list in final_trajectories.items():
        for traj_idx, trajectory_dict in enumerate(trajectories_list):
            if target_timestep in trajectory_dict: 
                joint_values = trajectory_dict[target_timestep]           
                features_for_timestep.append(joint_values)
                labels_for_timestep_full_names.append(object_name_full)
                instance_ids_for_timestep.append(f"{object_name_full}_traj{traj_idx}_ts{target_timestep}")

    if not features_for_timestep:
        print(f"Skipping timestep {target_timestep} for LR: No features.")
        counter +=1 
        continue
    
    X_timestep = np.array(features_for_timestep)
    label_encoder_full_names = LabelEncoder()
    y_encoded_full_names = label_encoder_full_names.fit_transform(labels_for_timestep_full_names)
    num_distinct_full_classes = len(label_encoder_full_names.classes_)

    label_counts_full_names = Counter(y_encoded_full_names)

   
    X_train, X_test, y_train_full_encoded, y_test_full_encoded, \
    train_labels_original, test_set_full_names_original, \
    train_instance_ids, test_instance_ids = train_test_split(
        X_timestep, y_encoded_full_names, labels_for_timestep_full_names, instance_ids_for_timestep,
        test_size=0.2, random_state=42, stratify=y_encoded_full_names
        )
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    actual_types_test = []
    unique_actual_types_in_test = []
    actual_types_test = [get_object_type(name) for name in test_set_full_names_original]
    unique_actual_types_in_test = sorted(list(set(actual_types_test)))

    y_pred_lr_full_encoded = [] 

    model_lr = LogisticRegression(random_state=42, multi_class='auto', max_iter=5000, **most_common_params_dict)
    model_lr.fit(X_train_scaled, y_train_full_encoded)
    
    y_pred_lr_train_encoded = model_lr.predict(X_train_scaled) 
    all_csv_data[counter]['LR_Train_Accuracy'] = accuracy_score(y_train_full_encoded, y_pred_lr_train_encoded)
    print(f"Timestep {target_timestep}: Logistic Regression Train Accuracy = {all_csv_data[counter]['LR_Train_Accuracy']:.4f}")

    y_pred_lr_full_encoded = model_lr.predict(X_test_scaled) 
    all_csv_data[counter]['LR_Overall_Test_Accuracy'] = accuracy_score(y_test_full_encoded, y_pred_lr_full_encoded)
    print(f"Timestep {target_timestep}: Logistic Regression Overall Test Accuracy = {all_csv_data[counter]['LR_Overall_Test_Accuracy']:.4f}")


    predicted_original_labels_lr = label_encoder_full_names.inverse_transform(y_pred_lr_full_encoded)
    for i in range(len(y_test_full_encoded)):
        lr_individual_predictions.append({
            'Timestep': target_timestep,
            'Instance_ID': test_instance_ids[i],
            'True_Label_Original': test_set_full_names_original[i],
            'LR_Predicted_Label_Original': predicted_original_labels_lr[i],
            'True_Label_Encoded': y_test_full_encoded[i],
            'LR_Predicted_Label_Encoded': y_pred_lr_full_encoded[i]
        })

    unique_test_labels_enc = [l for l in np.unique(y_test_full_encoded) if l < len(label_encoder_full_names.classes_)]
    filtered_classes = label_encoder_full_names.classes_[unique_test_labels_enc]
    report = classification_report(y_test_full_encoded, y_pred_lr_full_encoded, labels=unique_test_labels_enc, target_names=filtered_classes, output_dict=True, zero_division=0)
    for fon_name in all_full_object_names_global_csv:
        if fon_name in report:
            clean_name = fon_name.replace('/', '_').replace('-', '_')
            all_csv_data[counter][f'LR_Test_Accuracy_Object_{clean_name}'] = report[fon_name].get('recall', np.nan)

    counter += 1

######################################### MLP HYPERPARAMETER SEARCH  #########################################

counter = 0

param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.005], 
}

# base model
base_mlp_for_grid = MLPClassifier(random_state=42, max_iter=2000)

best_mlp_params_list = []
most_common_params_dict_mlp = {} 

for target_timestep in range(1, max_timesteps + 1, search_freq):
    print(f"\nProcessing timestep: {target_timestep}")

    features_for_timestep = []
    labels_for_timestep_full_names = []
    for object_name_full, trajectories_list in final_trajectories.items():
        for trajectory_dict in trajectories_list:
            joint_values = trajectory_dict[target_timestep]
            features_for_timestep.append(joint_values)
            labels_for_timestep_full_names.append(object_name_full)
    
    X_timestep = np.array(features_for_timestep)
    label_encoder_full_names = LabelEncoder()
    y_encoded_full_names = label_encoder_full_names.fit_transform(labels_for_timestep_full_names)
    num_distinct_full_classes = len(label_encoder_full_names.classes_)

    label_counts_full_names = Counter(y_encoded_full_names)
    
    X_train, X_test, y_train_full_encoded, y_test_full_encoded, _, test_set_full_names_original = train_test_split(
        X_timestep, y_encoded_full_names, labels_for_timestep_full_names,
        test_size=0.2, random_state=42, stratify=y_encoded_full_names
        )
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    actual_types_test = []
    unique_actual_types_in_test = []
    actual_types_test = [get_object_type(name) for name in test_set_full_names_original]
    unique_actual_types_in_test = sorted(list(set(actual_types_test)))

    y_pred_mlp_full_encoded = [] 
    
    #grid search
    grid_search_mlp = GridSearchCV(estimator=base_mlp_for_grid, 
                                   param_grid=param_grid_mlp, 
                                   cv=3,  
                                   scoring='accuracy', 
                                   n_jobs=-1, 
                                   verbose=0) #
    
    grid_search_mlp.fit(X_train_scaled, y_train_full_encoded)

    best_mlp_params_list.append(grid_search_mlp.best_params_)
    print(f"Timestep {target_timestep}: Best MLP Params: {grid_search_mlp.best_params_}")
    print(f"Timestep {target_timestep}: Best MLP CV Score on Train Data: {grid_search_mlp.best_score_:.4f}")

    model_mlp = grid_search_mlp.best_estimator_ 

    y_pred_mlp_train_encoded = model_mlp.predict(X_train_scaled) 
    all_csv_data[counter]['MLP_Train_Accuracy'] = accuracy_score(y_train_full_encoded, y_pred_mlp_train_encoded)
    print(f"Timestep {target_timestep}: MLP Train Accuracy (Best Model) = {all_csv_data[counter]['MLP_Train_Accuracy']:.4f}")

    y_pred_mlp_full_encoded = model_mlp.predict(X_test_scaled) 
    all_csv_data[counter]['MLP_Overall_Test_Accuracy'] = accuracy_score(y_test_full_encoded, y_pred_mlp_full_encoded)
    print(f"Timestep {target_timestep}: MLP Overall Test Accuracy (Best Model) = {all_csv_data[counter]['MLP_Overall_Test_Accuracy']:.4f}")

    unique_test_labels_enc = [l for l in np.unique(y_test_full_encoded) if l < len(label_encoder_full_names.classes_)]
    filtered_classes = label_encoder_full_names.classes_[unique_test_labels_enc]
    report = classification_report(y_test_full_encoded, y_pred_mlp_full_encoded, labels=unique_test_labels_enc, target_names=filtered_classes, output_dict=True, zero_division=0)
    for fon_name in all_full_object_names_global_csv:
        if fon_name in report:
            clean_name = fon_name.replace('/', '_').replace('-', '_')
            all_csv_data[counter][f'MLP_Test_Accuracy_Object_{clean_name}'] = report[fon_name].get('recall', np.nan)

    counter += 1

# determine best hyperparameters
if best_mlp_params_list:

    param_tuples_mlp = []
    for params in best_mlp_params_list:
        sorted_params = []
        for k, v in sorted(params.items()):
            if k == 'hidden_layer_sizes':
                sorted_params.append((k, tuple(v) if isinstance(v, list) else v ))
            else:
                sorted_params.append((k, v))
        param_tuples_mlp.append(tuple(sorted_params))

    param_counts_mlp = Counter(param_tuples_mlp)
    if param_counts_mlp: 
        most_common_params_tuple_mlp = param_counts_mlp.most_common(1)[0][0]
        most_common_params_dict_mlp = dict(most_common_params_tuple_mlp)
        print("\n--- Most Frequent Best MLP Hyperparameters Overall ---")
        print(f"Parameters: {most_common_params_dict_mlp}")
        print(f"Frequency: {param_counts_mlp.most_common(1)[0][1]} out of {len(best_mlp_params_list)} timesteps processed.")
    else:
        print("\nNo best MLP hyperparameters found to determine the most frequent.")
else:
    print("\nNo MLP hyperparameter search results to analyze.")

######################################### TRAIN MLP ##########################################

counter = 0

print('going to train the MLP')

for target_timestep_mlp in range(1, max_timesteps+1):
    print(f"\nProcessing timestep: {target_timestep_mlp}")

    features_for_timestep = []
    labels_for_timestep_full_names = []
    instance_ids_for_timestep = []

    for object_name_full, trajectories_list in final_trajectories.items():
        for traj_idx, trajectory_dict in enumerate(trajectories_list):
            if target_timestep_mlp in trajectory_dict: 
                joint_values = trajectory_dict[target_timestep_mlp]
                features_for_timestep.append(joint_values)
                labels_for_timestep_full_names.append(object_name_full)
                instance_ids_for_timestep.append(f"{object_name_full}_traj{traj_idx}_ts{target_timestep_mlp}")

    if not features_for_timestep:
        print(f"Skipping timestep {target_timestep_mlp} for MLP: No features.")
        counter += 1

        continue
    
    X_timestep = np.array(features_for_timestep)
    label_encoder_full_names = LabelEncoder()
    y_encoded_full_names = label_encoder_full_names.fit_transform(labels_for_timestep_full_names)
    num_distinct_full_classes = len(label_encoder_full_names.classes_)

    label_counts_full_names = Counter(y_encoded_full_names)
    

    X_train, X_test, y_train_full_encoded, y_test_full_encoded, \
    train_labels_original, test_set_full_names_original, \
    train_instance_ids, test_instance_ids = train_test_split(
        X_timestep, y_encoded_full_names, labels_for_timestep_full_names, instance_ids_for_timestep,
        test_size=0.2, random_state=42, stratify=y_encoded_full_names
        )
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    actual_types_test = []
    unique_actual_types_in_test = []
    actual_types_test = [get_object_type(name) for name in test_set_full_names_original]
    unique_actual_types_in_test = sorted(list(set(actual_types_test)))

    y_pred_mlp_full_encoded = [] 

    model_mlp = MLPClassifier(random_state=42, max_iter=2000, **most_common_params_dict_mlp) 
    model_mlp.fit(X_train_scaled, y_train_full_encoded)

    y_pred_mlp_train_encoded = model_mlp.predict(X_train_scaled) 

    all_csv_data[counter]['MLP_Train_Accuracy'] = accuracy_score(y_train_full_encoded, y_pred_mlp_train_encoded)


    y_pred_mlp_full_encoded = model_mlp.predict(X_test_scaled) 
    
    all_csv_data[counter]['MLP_Overall_Test_Accuracy'] = accuracy_score(y_test_full_encoded, y_pred_mlp_full_encoded)
    
    print(f"Timestep {target_timestep_mlp}: MLP Overall Test Accuracy = {accuracy_score(y_test_full_encoded, y_pred_mlp_full_encoded):.4f}")

    # Save MLP predictions
    predicted_original_labels_mlp = label_encoder_full_names.inverse_transform(y_pred_mlp_full_encoded)
    for i in range(len(y_test_full_encoded)):
        mlp_individual_predictions.append({
            'Timestep': target_timestep_mlp,
            'Instance_ID': test_instance_ids[i],
            'True_Label_Original': test_set_full_names_original[i],
            'MLP_Predicted_Label_Original': predicted_original_labels_mlp[i],
            'True_Label_Encoded': y_test_full_encoded[i],
            'MLP_Predicted_Label_Encoded': y_pred_mlp_full_encoded[i]
        })

    unique_test_labels_enc = [l for l in np.unique(y_test_full_encoded) if l < len(label_encoder_full_names.classes_)]
    filtered_classes = label_encoder_full_names.classes_[unique_test_labels_enc]
    report = classification_report(y_test_full_encoded, y_pred_mlp_full_encoded, labels=unique_test_labels_enc, target_names=filtered_classes, output_dict=True, zero_division=0)
    for fon_name in all_full_object_names_global_csv:
        if fon_name in report:
            clean_name = fon_name.replace('/', '_').replace('-', '_')
            all_csv_data[counter][f'MLP_Test_Accuracy_Object_{clean_name}'] = report[fon_name].get('recall', np.nan)
    
    counter += 1

################################## SAVE RESULTS TO CSV FOR FURTHER ANALYSIS ###########################################

results_df_csv = pd.DataFrame(all_csv_data)

ordered_csv_columns = ['Timestep', 'LR_Train_Accuracy', 'MLP_Train_Accuracy', 'LR_Overall_Test_Accuracy', 'MLP_Overall_Test_Accuracy']

object_specific_lr_cols = []
object_specific_mlp_cols = []
for name in all_full_object_names_global_csv:
    clean_name = name.replace('/', '_').replace('-', '_')
    col_lr = f'LR_Test_Accuracy_Object_{clean_name}'
    col_mlp = f'MLP_Test_Accuracy_Object_{clean_name}'
    object_specific_lr_cols.append(col_lr)
    object_specific_mlp_cols.append(col_mlp)

ordered_csv_columns.extend(sorted(object_specific_lr_cols))
ordered_csv_columns.extend(sorted(object_specific_mlp_cols))
        
final_csv_columns = [col for col in ordered_csv_columns if col in results_df_csv.columns]
remaining_csv_cols = sorted([col for col in results_df_csv.columns if col not in final_csv_columns])
final_csv_columns.extend(remaining_csv_cols)

results_df_csv = results_df_csv[final_csv_columns] 

csv_output_filename = 'training_and_object_test_accuracies_randomized_state_model.csv'
results_df_csv.to_csv(csv_output_filename, index=False, float_format='%.4f')
print(f"\nTraining and object-specific test accuracies saved to {csv_output_filename}")

df_lr_preds = pd.DataFrame(lr_individual_predictions)
df_mlp_preds = pd.DataFrame(mlp_individual_predictions)

if not df_lr_preds.empty and not df_mlp_preds.empty:
    df_all_individual_preds = pd.merge(
        df_lr_preds,
        df_mlp_preds.drop(columns=['True_Label_Original', 'True_Label_Encoded']), 
        on=['Timestep', 'Instance_ID'],
        how='outer'
    )
elif not df_lr_preds.empty:
    df_all_individual_preds = df_lr_preds
elif not df_mlp_preds.empty:
    df_all_individual_preds = df_mlp_preds
else:
    df_all_individual_preds = pd.DataFrame()

if not df_all_individual_preds.empty:
    detailed_predictions_filename = 'detailed_instance_predictions.csv'
    df_all_individual_preds.to_csv(detailed_predictions_filename, index=False, float_format='%.4f')
    print(f"Detailed instance-level predictions saved to {detailed_predictions_filename}")
else:
    print("No detailed individual predictions were recorded to save.")