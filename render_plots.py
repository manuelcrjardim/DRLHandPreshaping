import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
from collections import defaultdict

# Configuration
SUMMARY_CSV_PATH = 'training_and_object_test_accuracies_randomized_state_model_GPU.csv'
DETAILED_CSV_PATH = 'detailed_instance_predictions.csv'
OUTPUT_DIR = 'analysis_plots'
COMMON_FIG_SIZE = (16, 8)
CM_FIG_SIZE = (10, 8)

# Font sizes
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 12
ANNOT_FONT_SIZE = 14

# Chance level
CHANCE_LEVEL = 1/22
CHANCE_LINE_COLOR = 'red'
CHANCE_LINE_STYLE = '--'
CHANCE_LINE_WIDTH = 2
CHANCE_LINE_ALPHA = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_object_type(full_object_name):
    parts = str(full_object_name).split('/')
    type_part = parts[1].split('-')
    return type_part[0]

# Load data
df_summary = pd.read_csv(SUMMARY_CSV_PATH)
df_detailed = pd.read_csv(DETAILED_CSV_PATH)

# Plot average accuracies
def plot_average_accuracies(df_summary, output_dir):
    df_filtered_summary = df_summary[((df_summary['Timestep'] - 1) % 5 == 0) | (df_summary['Timestep'] == df_summary['Timestep'].min()) | (df_summary['Timestep'] == df_summary['Timestep'].max())].copy()
    
    plt.figure(figsize=COMMON_FIG_SIZE)
    plt.plot(df_filtered_summary['Timestep'], df_filtered_summary['LR_Overall_Test_Accuracy'], label='LR Overall Test Accuracy', marker='o', linestyle='-')
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Timestep', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_overall_accuracy_sampled.png'))
    plt.close()

    plt.figure(figsize=COMMON_FIG_SIZE)
    plt.plot(df_filtered_summary['Timestep'], df_filtered_summary['MLP_Overall_Test_Accuracy'], label='MLP Overall Test Accuracy', marker='o', linestyle='--')
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Timestep', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_overall_accuracy_sampled.png'))
    plt.close()

# Plot accuracy by object type
def plot_accuracy_by_object_type(df_detailed, model_prefix, model_name, output_dir):
    df_model_preds = df_detailed.dropna(subset=[f'{model_prefix}_Predicted_Label_Original', 'True_Label_Original']).copy()
    df_model_preds['True_Object_Type'] = df_model_preds['True_Label_Original'].apply(get_object_type)
    df_model_preds['Predicted_Object_Type'] = df_model_preds[f'{model_prefix}_Predicted_Label_Original'].apply(get_object_type)

    accuracies_by_type_timestep = defaultdict(list)
    original_timesteps = sorted(df_model_preds['Timestep'].unique())
    
    sampled_timesteps = original_timesteps
    if len(original_timesteps) > 10:
        sampled_timesteps = [ts for ts in original_timesteps if ((ts - 1) % 5 == 0) or ts == original_timesteps[0] or ts == original_timesteps[-1]]
        sampled_timesteps = sorted(list(set(sampled_timesteps)))

    object_types = sorted(df_model_preds['True_Object_Type'].unique())

    for ts in sampled_timesteps: 
        df_ts = df_model_preds[df_model_preds['Timestep'] == ts]
        for obj_type in object_types:
            df_type_ts = df_ts[df_ts['True_Object_Type'] == obj_type]   
            if not df_type_ts.empty:
                correct_predictions = (df_type_ts['Predicted_Object_Type'] == obj_type).sum()
                acc = correct_predictions / len(df_type_ts)
                accuracies_by_type_timestep[obj_type].append({'Timestep': ts, 'Accuracy': acc})
            else:
                accuracies_by_type_timestep[obj_type].append({'Timestep': ts, 'Accuracy': np.nan})
    
    plt.figure(figsize=COMMON_FIG_SIZE)
    for obj_type, acc_data in accuracies_by_type_timestep.items():
        df_plot = pd.DataFrame(acc_data)
        plt.plot(df_plot['Timestep'], df_plot['Accuracy'], label=f'Type: {obj_type}', marker='.')
    
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Timestep', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE, ncol=2)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    plt.savefig(os.path.join(output_dir, f'{model_prefix.lower()}_accuracy_by_object_type_final.png'))
    plt.close()

# Plot MLP confusion matrices
def plot_mlp_confusion_matrices(df_detailed, timesteps_for_cm, output_dir):
    df_detailed_cm = df_detailed.copy()
    df_detailed_cm['True_Object_Type'] = df_detailed_cm['True_Label_Original'].apply(get_object_type)
    df_detailed_cm['MLP_Predicted_Object_Type'] = df_detailed_cm['MLP_Predicted_Label_Original'].apply(get_object_type)
    
    all_object_types_for_labels = sorted(list(set(
        list(df_detailed_cm['True_Object_Type'].dropna().unique()) + 
        list(df_detailed_cm['MLP_Predicted_Object_Type'].dropna().unique())
    )))

    for ts in timesteps_for_cm:
        df_ts_original = df_detailed_cm[df_detailed_cm['Timestep'] == ts]
        df_mlp_cm_data_unbalanced = df_ts_original.dropna(subset=['True_Object_Type', 'MLP_Predicted_Object_Type'])
        
        counts_per_type_mlp = df_mlp_cm_data_unbalanced['True_Object_Type'].value_counts()
        min_samples_mlp = counts_per_type_mlp.min()
        
        df_mlp_cm_data_balanced = df_mlp_cm_data_unbalanced.groupby('True_Object_Type', group_keys=False).apply(
            lambda x: x.sample(min_samples_mlp, random_state=42)
        ).reset_index(drop=True)
        
        cm_mlp = confusion_matrix(df_mlp_cm_data_balanced['True_Object_Type'], df_mlp_cm_data_balanced['MLP_Predicted_Object_Type'], labels=all_object_types_for_labels)
        plt.figure(figsize=CM_FIG_SIZE)
        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=all_object_types_for_labels, yticklabels=all_object_types_for_labels,
                    annot_kws={"size": ANNOT_FONT_SIZE})
        plt.xlabel('Predicted Object Type', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('True Object Type', fontsize=LABEL_FONT_SIZE)
        plt.xticks(rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
        plt.yticks(rotation=0, fontsize=TICK_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mlp_type_confusion_matrix_ts{ts}_balanced.png'))
        plt.close()

# Combined overall accuracy
def plot_combined_overall_accuracy(df_summary, output_dir):
    plt.figure(figsize=COMMON_FIG_SIZE)
    plt.plot(df_summary['Timestep'], df_summary['LR_Overall_Test_Accuracy'], label='LR Overall Test Accuracy', marker='o', linestyle='-')
    plt.plot(df_summary['Timestep'], df_summary['MLP_Overall_Test_Accuracy'], label='MLP Overall Test Accuracy', marker='x', linestyle='--')
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Timestep', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_vs_mlp_overall_accuracy.png'))
    plt.close()

# Mean accuracy per type bars
def plot_mean_accuracy_per_type_bars(df_detailed, model_prefix, model_name, output_dir):
    df_model_preds = df_detailed.dropna(subset=[f'{model_prefix}_Predicted_Label_Original', 'True_Label_Original']).copy()
    df_model_preds['True_Object_Type'] = df_model_preds['True_Label_Original'].apply(get_object_type)
    df_model_preds['Predicted_Object_Type'] = df_model_preds[f'{model_prefix}_Predicted_Label_Original'].apply(get_object_type)
    
    type_accuracies = []
    valid_true_types = df_model_preds['True_Object_Type'].unique()

    for obj_type in sorted(valid_true_types):
        df_type = df_model_preds[df_model_preds['True_Object_Type'] == obj_type]
        correct_predictions = (df_type['Predicted_Object_Type'] == obj_type).sum()
        total_predictions = len(df_type)
        acc = correct_predictions / total_predictions
        type_accuracies.append({'Object_Type': obj_type, 'Mean_Accuracy': acc})
    
    df_plot = pd.DataFrame(type_accuracies)
    df_plot = df_plot.sort_values(by='Mean_Accuracy', ascending=False)

    plt.figure(figsize=(max(12, len(df_plot)*0.6), 7))
    sns.barplot(x='Object_Type', y='Mean_Accuracy', data=df_plot, palette='viridis')
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Object Type', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Mean Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_prefix.lower()}_mean_accuracy_per_type_bars.png'))
    plt.close()

# Average accuracy by object type
def plot_average_accuracy_by_object_type(df_detailed, object_types_to_plot, output_dir):
    df_copy = df_detailed.copy()
    df_copy['True_Object_Type'] = df_copy['True_Label_Original'].apply(get_object_type)

    avg_accuracies_list = []

    for obj_type in object_types_to_plot:
        type_data = df_copy[df_copy['True_Object_Type'] == obj_type]
        
        lr_pred_type = type_data['LR_Predicted_Label_Original'].apply(get_object_type)
        lr_acc = (lr_pred_type == obj_type).mean()
        avg_accuracies_list.append({'Object_Type': obj_type, 'Model': 'LR', 'Average_Accuracy': lr_acc})
        
        mlp_pred_type = type_data['MLP_Predicted_Label_Original'].apply(get_object_type)
        mlp_acc = (mlp_pred_type == obj_type).mean()
        avg_accuracies_list.append({'Object_Type': obj_type, 'Model': 'MLP', 'Average_Accuracy': mlp_acc})
        
    df_plot = pd.DataFrame(avg_accuracies_list)
    plt.figure(figsize=COMMON_FIG_SIZE)
    sns.barplot(data=df_plot, x='Object_Type', y='Average_Accuracy', hue='Model', palette={'LR': 'skyblue', 'MLP': 'lightcoral'})
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Object Type', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Overall Average Accuracy', fontsize=LABEL_FONT_SIZE)
    plt.xticks(rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Model', fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_accuracy_by_selected_object_type_bars.png'))
    plt.close()

# Selected types accuracy comparison
def plot_selected_types_accuracy_comparison(df_detailed, object_types_to_plot, output_dir):
    df_plot_data_list = []
    df_copy = df_detailed.copy()
    df_copy['True_Object_Type'] = df_copy['True_Label_Original'].apply(get_object_type)
    df_copy['LR_Predicted_Object_Type'] = df_copy['LR_Predicted_Label_Original'].apply(get_object_type)
    df_copy['MLP_Predicted_Object_Type'] = df_copy['MLP_Predicted_Label_Original'].apply(get_object_type)

    df_relevant_types = df_copy[df_copy['True_Object_Type'].isin(object_types_to_plot)]
    original_timesteps = sorted(df_relevant_types['Timestep'].unique())
    sampled_timesteps = original_timesteps
    if len(original_timesteps) > 10:
        sampled_timesteps = [ts for ts in original_timesteps if ((ts - 1) % 5 == 0) or ts == original_timesteps[0] or ts == original_timesteps[-1]]
        sampled_timesteps = sorted(list(set(sampled_timesteps)))

    for obj_type in object_types_to_plot:
        df_type_specific = df_relevant_types[df_relevant_types['True_Object_Type'] == obj_type]

        for ts in sampled_timesteps:
            df_ts_type = df_type_specific[df_type_specific['Timestep'] == ts]
            
            total_instances = len(df_ts_type)
            correct_lr = (df_ts_type['LR_Predicted_Object_Type'] == obj_type).sum()
            accuracy_lr = correct_lr / total_instances if total_instances > 0 else np.nan
            df_plot_data_list.append({'Timestep': ts, 'Object_Type': obj_type, 'Model': 'LR', 'Accuracy': accuracy_lr})
            
            correct_mlp = (df_ts_type['MLP_Predicted_Object_Type'] == obj_type).sum()
            accuracy_mlp = correct_mlp / total_instances if total_instances > 0 else np.nan
            df_plot_data_list.append({'Timestep': ts, 'Object_Type': obj_type, 'Model': 'MLP', 'Accuracy': accuracy_mlp})

    df_for_plotting = pd.DataFrame(df_plot_data_list)
    plt.figure(figsize=COMMON_FIG_SIZE)
    sns.lineplot(data=df_for_plotting, x='Timestep', y='Accuracy', hue='Object_Type', style='Model', marker='o', dashes=True)
    plt.axhline(y=CHANCE_LEVEL, color=CHANCE_LINE_COLOR, linestyle=CHANCE_LINE_STYLE, 
                linewidth=CHANCE_LINE_WIDTH, alpha=CHANCE_LINE_ALPHA, label=f'Chance Level ({CHANCE_LEVEL:.2f})')
    plt.xlabel('Timestep', fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Accuracy (Correct Type Prediction)', fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(title='Object Type & Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(os.path.join(output_dir, 'selected_types_accuracy_comparison_lines.png'))
    plt.close()

# Main execution
plot_average_accuracies(df_summary, OUTPUT_DIR)
plot_accuracy_by_object_type(df_detailed.copy(), 'LR', 'Logistic Regression', OUTPUT_DIR)
plot_accuracy_by_object_type(df_detailed.copy(), 'MLP', 'MLP', OUTPUT_DIR)

# Select timesteps for confusion matrices
timesteps_for_cm = [1, 25, 45, 130, 200]
available_timesteps = sorted(df_detailed['Timestep'].unique())
timesteps_for_cm = [ts for ts in timesteps_for_cm if ts in available_timesteps]

plot_mlp_confusion_matrices(df_detailed.copy(), timesteps_for_cm, OUTPUT_DIR)
plot_combined_overall_accuracy(df_summary, OUTPUT_DIR)
plot_mean_accuracy_per_type_bars(df_detailed.copy(), 'LR', 'Logistic Regression', OUTPUT_DIR)
plot_mean_accuracy_per_type_bars(df_detailed.copy(), 'MLP', 'MLP', OUTPUT_DIR)

# Get object types for visualization
all_present_types = sorted([
    ptype for ptype in df_detailed['True_Label_Original'].apply(get_object_type).unique()
])
object_types_to_visualize = all_present_types[:3]

plot_average_accuracy_by_object_type(df_detailed.copy(), object_types_to_visualize, OUTPUT_DIR)
plot_selected_types_accuracy_comparison(df_detailed.copy(), object_types_to_visualize, OUTPUT_DIR)

print(f"Plots saved in: {os.path.abspath(OUTPUT_DIR)}")