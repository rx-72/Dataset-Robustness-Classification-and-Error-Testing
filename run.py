import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
from etl import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

with open("data-params.json", "r") as f:
  params = json.load(f)

METRICS = {
    "accuracy": accuracy
}

def run_discretization_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, robustness_radius, bin_numbers, max_uncertain_pct=10, maximize=True):
    def plot_heatmap(ax, heatmap_data, x_labels, y_labels, title):
        heatmap = ax.imshow(heatmap_data, cmap=cmap, interpolation='nearest', 
                        aspect='auto', alpha=0.8, vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    
        # Add white lines by adjusting the linewidth for minor ticks
        ax.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
        # Remove external boundaries
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if heatmap_data[i][j] == 100:
                    text = ax.text(j, i, '100', ha='center', va='center', color='black')
                elif heatmap_data[i][j] == 0:
                    text = ax.text(j, i, '0', ha='center', va='center', color='black')
                else:
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', color='black')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Percentage of Uncertain Data', fontsize=12)
        ax.set_ylabel('Uncertain Radius (%)', fontsize=12)
        return heatmap

    def select_best_feature(X, y, method="correlation"):
        if method == "correlation":
            correlations = X.corrwith(y)
            best_feature = correlations.abs().idxmax()
        return best_feature

    def discretize_and_sample(X_train, feature, thresholds, total_samples, num_bins):
        feature_values = X_train[feature]  # Extract the column of interest
        min_val, max_val = feature_values.min(), feature_values.max()

        # Ensure the values are within the valid range of bins (clip out-of-range values)
        feature_values = feature_values.clip(lower=min_val, upper=max_val)

        # Dynamically calculate bin edges based on the feature values range
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_counts, bin_labels = pd.cut(feature_values, bins=bin_edges, labels=False, retbins=True)

        selected_bins_by_threshold = []  # Combined threshold list
        for thresh in thresholds:  # Iterate over age thresholds
            selected_bins = []  # New bin list for that respective threshold
            for bin_idx in range(num_bins):  # For each bin
                bin_lower = bin_edges[bin_idx]  # Get lower bin threshold at bin_idx
                bin_upper = bin_edges[bin_idx + 1]  # Get upper bin threshold at bin_idx
                if thresh[0] <= bin_lower and thresh[1] >= bin_upper:  # If bin meets threshold requirements
                    selected_bins.append(bin_idx)
            selected_bins_by_threshold.append(selected_bins)  # Add threshold bin list to combined threshold list

        sampled_indices = set()

        for threshold_idx, selected_bins in enumerate(selected_bins_by_threshold):
            bin_freqs = feature_values.groupby(bin_counts).size()  # Calculate bin frequency of each bin for this threshold
        
            # Reindex to ensure all bins are accounted for, including those with zero frequency
            bin_freqs = bin_freqs.reindex(range(num_bins), fill_value=0)

            bin_priority = sorted([(bin_idx, bin_freqs[bin_idx])  # Sort by frequency of bin, decreasing
                           for bin_idx in selected_bins], 
                          key=lambda x: -x[1]) 

            for bin_idx, _ in bin_priority:  # Enumerate over the bins ordered by frequency for this threshold
                bin_indices = feature_values[bin_counts == bin_idx].index  # Get the indices
                needed = total_samples - len(sampled_indices)  # Get however many values are still needed to grab
            
                for idx in bin_indices:  # Iterate over indices
                    if len(sampled_indices) < total_samples:
                        sampled_indices.add(idx)
                    else:
                        break
                if len(sampled_indices) >= total_samples:
                    break

            if len(sampled_indices) >= total_samples:  # End early if total samples needed is met
                break
            
        return list(sampled_indices), bin_edges

    def find_best_bin_size(X_train, y_train, X_test, y_test, feature, thresholds, ratio, robustness_radius, uncertain_percentage):

        label_range = (y_train.max() - y_train.min())
        uncertain_radius = ratio * label_range
        min_robustness_ratio = float('inf')
        best_bins = None
        best_result = None
    
        # Wrapping the range with tqdm to display a progress bar
        for num_bins in tqdm(range(5, 51), desc='Testing bin sizes'):  # Progress bar with a description
            sampled_indices, bin_edges = discretize_and_sample(X_train, feature, thresholds, 
                                                           total_samples=int(uncertain_percentage * len(X_train)), 
                                                           num_bins=num_bins)
        
            boundary_indices = np.array(sampled_indices)
            robustness_ratio = compute_robustness_ratio_sensitive_label_error(
                X_train, y_train, X_test, y_test,
                uncertain_num=int(uncertain_percentage * len(X_train)),
                boundary_indices=boundary_indices,
                uncertain_radius=uncertain_radius,
                robustness_radius=robustness_radius,
                interval=False
            )
        
            if robustness_ratio < min_robustness_ratio:
                min_robustness_ratio = robustness_ratio
                best_bins = num_bins
                best_result = (bin_edges, robustness_ratio)
            elif robustness_ratio == min_robustness_ratio:
                if num_bins > best_bins:
                    best_bins = num_bins
                    best_result = (bin_edges, robustness_ratio)
    
        return best_bins, best_result
    
    def get_positive_paths(tree, feature_names, node=0, depth=0, conditions=None, results=None, min_positive_ratio=0.5):
        if conditions is None:
            conditions = {}
        if results is None:
            results = []

        left_child = tree.children_left[node]
        right_child = tree.children_right[node]
        threshold = tree.threshold[node]
        feature = tree.feature[node]

        # Count samples in this node
        sample_count = int(tree.n_node_samples[node])
        positive_count = int(tree.value[node][0, 1]) if tree.n_outputs == 1 else int(tree.value[node][0][1])
        negative_count = int(tree.value[node][0, 0]) if tree.n_outputs == 1 else int(tree.value[node][0][0])

        # Calculate the positive ratio for this node
        positive_ratio = positive_count / sample_count if sample_count > 0 else 0

        # If it's a leaf or qualifies as a 'positive node' by ratio, store the path
        if (left_child == _tree.TREE_LEAF and right_child == _tree.TREE_LEAF) or positive_ratio >= min_positive_ratio:
            path_conditions = {}
            for feat, bounds in conditions.items():
                lower_bound = bounds.get('lower', 0)
                upper_bound = bounds.get('upper', feature_max_values.get(feat, '∞'))  # Use the max value for the feature
                path_conditions[feat] = (lower_bound, upper_bound)
        
        # Only store if there are significant positives
            if positive_count > 0:  # Ensure that there's at least one positive sample
                results.append((positive_count, sample_count, path_conditions, positive_ratio, depth))

        # Update bounds for the current feature in conditions and recurse
        feature_name = feature_names[feature] if feature != _tree.TREE_UNDEFINED else None
        if left_child != _tree.TREE_LEAF and feature_name:
            # Left child represents the <= threshold split
            new_conditions = {k: v.copy() for k, v in conditions.items()}
            new_conditions.setdefault(feature_name, {}).update({'upper': threshold})
            get_positive_paths(tree, feature_names, left_child, depth + 1, new_conditions, results, min_positive_ratio)

        if right_child != _tree.TREE_LEAF and feature_name:
            # Right child represents the > threshold split
            new_conditions = {k: v.copy() for k, v in conditions.items()}
            new_conditions.setdefault(feature_name, {}).update({'lower': threshold})
            get_positive_paths(tree, feature_names, right_child, depth + 1, new_conditions, results, min_positive_ratio)

        # Print and store paths after completing all nodes, if we are at the root node
        if node == 0:
            # Sort results first by depth (root to leaf), then by positive ratio, and then by positive count
            top_results = sorted(results, key=lambda x: (x[4], x[3], x[0]), reverse=False)[:3]  # Prioritize by depth first
        
            # Store top thresholds
            top_thresholds = []
            for idx, (pos_count, total_count, conditions, pos_ratio, dep) in enumerate(top_results, start=1):
                top_thresholds.append(conditions)  # Save conditions (thresholds) for each scenario    
            return top_thresholds

    print("")
    print(f"Generating important indices based on RandomForest, MAE:")
    print("")

    best_feature = select_best_feature(X_train, y_train)
    lr = RandomForestRegressor
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, lr, mae, maximize=False)

    # Decision Tree research: 1% of the data
    array_indexes = np.zeros(len(X_train))
    perc = 0.1 * len(X_train)
    for i in range(0, len(X_train)):
        if i <= perc:
            index = boundary_indices[i]
            array_indexes[index] = 1

    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(X_train, array_indexes)

    feature_max_values = X_train.max()
    tree = clf.tree_
    feature_names = X_train.columns
    best_thresholds = get_positive_paths(tree, feature_names)
    thresholds = [thres[best_feature] for thres in best_thresholds if best_feature in thres]

    print("")
    print(f"Running ZORRO on RandomForest, MAE:")
    print("")
    
    boundary_indices, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bin_numbers[0])
    # Testing more extreme uncertain percentages
    robustness_dicts_rndFrst_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        robustness_radius = robustness_radius
        label_range = (y_train.max()-y_train.min())
        ratios = ratios
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                #print(uncertain_radius)
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_rndFrst_mae.append(robustness_dict)

    print("")
    print(f"Running ZORRO on naive method:")
    print("")
    
    robustness_dicts_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_naive.append(robustness_dict)

    print("")
    print(f"Running Meyer on RandomForest, MAE:")
    print("")
    
    robustness_dicts_interval_rndFrst_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_rndFrst_mae.append(robustness_dict_interval)

    print("")
    print(f"Running Meyer on naive method:")
    print("")
    
    robustness_dicts_interval_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_naive.append(robustness_dict_interval)

    print("")
    print(f"Generating important indices based on RandomForest, MSE:")
    print("")
    
    lr = RandomForestRegressor
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, lr, mse, maximize=False)

    # Decision Tree research: 1% of the data
    array_indexes = np.zeros(len(X_train))
    perc = 0.1 * len(X_train)
    for i in range(0, len(X_train)):
        if i <= perc:
            index = boundary_indices[i]
            array_indexes[index] = 1

    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(X_train, array_indexes)

    feature_max_values = X_train.max()
    tree = clf.tree_
    feature_names = X_train.columns
    best_thresholds = get_positive_paths(tree, feature_names)
    thresholds = [thres[best_feature] for thres in best_thresholds if best_feature in thres]

    print("")
    print(f"Running ZORRO on RandomForest, MSE:")
    print("")
    
    boundary_indices, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bin_numbers[1])
    robustness_dicts_rndFrst_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        robustness_radius = robustness_radius
        label_range = (y_train.max()-y_train.min())
        ratios = ratios
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                #print(uncertain_radius)
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_rndFrst_mse.append(robustness_dict)

    print("")
    print(f"Running Meyer on RandomForest, MSE:")
    print("")
    
    robustness_dicts_interval_rndFrst_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_rndFrst_mse.append(robustness_dict_interval)

    print("")
    print(f"Generating important indices based on LinearRegression, MAE:")
    print("")
    
    lr = LinearRegression
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, lr, mae, maximize=False)

    # Decision Tree research: 1% of the data
    array_indexes = np.zeros(len(X_train))
    perc = 0.1 * len(X_train)
    for i in range(0, len(X_train)):
        if i <= perc:
            index = boundary_indices[i]
            array_indexes[index] = 1

    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(X_train, array_indexes)

    feature_max_values = X_train.max()
    tree = clf.tree_
    feature_names = X_train.columns
    best_thresholds = get_positive_paths(tree, feature_names)
    thresholds = [thres[best_feature] for thres in best_thresholds if best_feature in thres]

    boundary_indices, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bin_numbers[2])

    print("")
    print(f"Running ZORRO on LinearRegression, MAE:")
    print("")
    
    robustness_dicts_linReg_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        robustness_radius = robustness_radius
        label_range = (y_train.max()-y_train.min())
        ratios = ratios
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                #print(uncertain_radius)
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_linReg_mae.append(robustness_dict)

    print("")
    print(f"Running Meyer on LinearRegression, MAE:")
    print("")
    
    robustness_dicts_interval_linReg_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_linReg_mae.append(robustness_dict_interval)

    print("")
    print(f"Generating important indices based on LinearRegression, MSE:")
    print("")
    
    lr = LinearRegression
    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, lr, mse, maximize=False)

    # Decision Tree research: 1% of the data
    array_indexes = np.zeros(len(X_train))
    perc = 0.1 * len(X_train)
    for i in range(0, len(X_train)):
        if i <= perc:
            index = boundary_indices[i]
            array_indexes[index] = 1

    clf = DecisionTreeClassifier(max_depth=None)
    clf.fit(X_train, array_indexes)

    feature_max_values = X_train.max()
    tree = clf.tree_
    feature_names = X_train.columns
    best_thresholds = get_positive_paths(tree, feature_names)
    thresholds = [thres[best_feature] for thres in best_thresholds if best_feature in thres]

    boundary_indices, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bin_numbers[3])

    print("")
    print(f"Running ZORRO on LinearRegression, MSE:")
    print("")
    
    robustness_dicts_linReg_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        robustness_radius = robustness_radius
        label_range = (y_train.max()-y_train.min())
        ratios = ratios
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                #print(uncertain_radius)
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_linReg_mse.append(robustness_dict)

    print("")
    print(f"Running Meyer on LinearRegression, MSE:")
    print("")
    
    robustness_dicts_interval_linReg_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_linReg_mse.append(robustness_dict_interval)

    print("Heatmaps:")

    # Create the heatmap plot with a 2x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(18, 18), dpi=200)

    # Define colormap
    cmap = plt.get_cmap("autumn_r")

    print("Formatting data for the heatmaps")
    
    df1 = sum([pd.DataFrame(robustness_dicts_interval_naive[i]).iloc[:, 2:] for i in range(5)])/5
    df2 = sum([pd.DataFrame(robustness_dicts_naive[i]).iloc[:, 2:] for i in range(5)])/5
    df3 = sum([pd.DataFrame(robustness_dicts_interval_rndFrst_mae[i]).iloc[:, 2:] for i in range(5)])/5  
    df4 = sum([pd.DataFrame(robustness_dicts_rndFrst_mae[i]).iloc[:, 2:] for i in range(5)])/5
    df5 = sum([pd.DataFrame(robustness_dicts_interval_rndFrst_mse[i]).iloc[:, 2:] for i in range(5)])/5
    df6 = sum([pd.DataFrame(robustness_dicts_rndFrst_mse[i]).iloc[:, 2:] for i in range(5)])/5
    df7 = sum([pd.DataFrame(robustness_dicts_interval_linReg_mae[i]).iloc[:, 2:] for i in range(5)])/5  
    df8 = sum([pd.DataFrame(robustness_dicts_linReg_mae[i]).iloc[:, 2:] for i in range(5)])/5
    df9 = sum([pd.DataFrame(robustness_dicts_interval_linReg_mse[i]).iloc[:, 2:] for i in range(5)])/5  
    df10 = sum([pd.DataFrame(robustness_dicts_linReg_mse[i]).iloc[:, 2:] for i in range(5)])/5

    print("Converting fractions to percentages")
    heatmap_data1 = df1.multiply(100).values
    heatmap_data2 = df2.multiply(100).values
    heatmap_data3 = df3.multiply(100).values
    heatmap_data4 = df4.multiply(100).values
    heatmap_data5 = df5.multiply(100).values
    heatmap_data6 = df6.multiply(100).values
    heatmap_data7 = df7.multiply(100).values
    heatmap_data8 = df8.multiply(100).values
    heatmap_data9 = df9.multiply(100).values
    heatmap_data10 = df10.multiply(100).values

    # Labels
    x_labels = df1.columns.tolist()
    y_labels = ratios

    # Plot each heatmap
    heatmaps = []
    heatmaps.append(plot_heatmap(axes[0, 0], heatmap_data1, x_labels, y_labels, 'Meyer et al. (Naive Approach)'))
    heatmaps.append(plot_heatmap(axes[0, 1], heatmap_data2, x_labels, y_labels, 'ZORRO (Naive Approach)'))
    heatmaps.append(plot_heatmap(axes[1, 0], heatmap_data9, x_labels, y_labels, 'Meyer et al. (LinReg, mse)'))
    heatmaps.append(plot_heatmap(axes[1, 1], heatmap_data10, x_labels, y_labels, 'ZORRO (LinReg, mse)'))
    heatmaps.append(plot_heatmap(axes[2, 0], heatmap_data7, x_labels, y_labels, 'Meyer et al. (LinReg, mae)'))
    heatmaps.append(plot_heatmap(axes[2, 1], heatmap_data8, x_labels, y_labels, 'ZORRO (LinReg, mae)'))
    heatmaps.append(plot_heatmap(axes[3, 0], heatmap_data5, x_labels, y_labels, 'Meyer et al. (RndFrst, mse)'))
    heatmaps.append(plot_heatmap(axes[3, 1], heatmap_data6, x_labels, y_labels, 'ZORRO (RndFrst, mse)'))
    heatmaps.append(plot_heatmap(axes[4, 0], heatmap_data3, x_labels, y_labels, 'Meyer et al. (RndFrst, mae)'))
    heatmaps.append(plot_heatmap(axes[4, 1], heatmap_data4, x_labels, y_labels, 'ZORRO (RndFrst, mae)'))

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            print(f"Axes[{i}, {j}] images: {ax.images}")

    # Adjust layout and add colorbar
    plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.1, left=0.1, right=0.9)
    cb = fig.colorbar(heatmaps[-1], ax=axes.ravel().tolist(), orientation='vertical', pad=0.02)
    cb.set_label('Robustness Ratio (%)', fontsize=12)
    
    plt.savefig(f"{output_dir}/{args.dataset}-Discretization-method-heatmap.pdf", bbox_inches='tight')
  
    print("")
    print("Discretization finished!")
    print("")

def run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, robustness_radius, max_uncertain_pct=10, maximize=True):
    def plot_heatmap(ax, heatmap_data, x_labels, y_labels, title):
        heatmap = ax.imshow(heatmap_data, cmap=cmap, interpolation='nearest', 
                        aspect='auto', alpha=0.8, vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    
        # Add white lines by adjusting the linewidth for minor ticks
        ax.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
        # Remove external boundaries
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if heatmap_data[i][j] == 100:
                    text = ax.text(j, i, '100', ha='center', va='center', color='black')
                elif heatmap_data[i][j] == 0:
                    text = ax.text(j, i, '0', ha='center', va='center', color='black')
                else:
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', color='black')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Percentage of Uncertain Data', fontsize=12)
        ax.set_ylabel('Uncertain Radius (%)', fontsize=12)
        return heatmap
    
    print("")
    print(f"Generating important indices based on Linear Regression, MSE:")
    print("")

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, LinearRegression, mse, maximize)
  

    print("")
    print(f"Running Leave One Out (using Linear Regression, MSE) on ZORRO.")
    print("")
    
    robustness_dicts_lr_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_lr_mse.append(robustness_dict)

    print("")
    print(f"Running Naive Approach on ZORRO.")
    print("")

    robustness_dicts_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
            
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_naive.append(robustness_dict)

    print()
    print(f"Running Leave One Out (using Linear Regression, MSE) on Meyer.")
    print()
    
    robustness_dicts_interval_lr_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_lr_mse.append(robustness_dict_interval)

    
    print("")
    print(f"Running Naive Approach on Meyer.")
    print("")

    robustness_dicts_interval_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_naive.append(robustness_dict_interval) 
    
    
    print("")
    print(f"Generating important indices based on Linear Regression, MAE:")
    print("")

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, LinearRegression, mae, maximize)

    print("")
    print(f"Running Leave One Out (using Linear Regression, MAE) on ZORRO.")
    print("")
    
    robustness_dicts_lr_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_lr_mae.append(robustness_dict)

    print()
    print(f"Running Leave One Out (using Linear Regression, MAE) on Meyer.")
    print()
    
    robustness_dicts_interval_lr_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_lr_mae.append(robustness_dict_interval)

    print("")
    print(f"Generating important indices based on RandomForestRegressor, MSE:")
    print("")

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, RandomForestRegressor, mse, maximize)

    print("")
    print(f"Running Leave One Out (using RandomForestRegressor, MSE) on ZORRO.")
    print("")
    
    robustness_dicts_rf_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_rf_mse.append(robustness_dict)

    print()
    print(f"Running Leave One Out (using RandomForestRegressor, MSE) on Meyer.")
    print()
    
    robustness_dicts_interval_rf_mse = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_rf_mse.append(robustness_dict_interval)

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, RandomForestRegressor, mae, maximize)

    print("")
    print(f"Running Leave One Out (using RandomForestRegressor, MAE) on ZORRO.")
    print("")
    
    robustness_dicts_rf_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_rf_mae.append(robustness_dict)

    print()
    print(f"Running Leave One Out (using RandomForestRegressor, MAE) on Meyer.")
    print()
    
    robustness_dicts_interval_rf_mae = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_rf_mae.append(robustness_dict_interval)  


    print("Heatmaps:")

    # Create the heatmap plot with a 2x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(18, 18), dpi=200)

    # Define colormap
    cmap = plt.get_cmap("autumn_r")

    print("Formatting data for the heatmaps")
    
    df1 = sum([pd.DataFrame(robustness_dicts_interval_naive[i]).iloc[:, 2:] for i in range(5)])/5
    df2 = sum([pd.DataFrame(robustness_dicts_interval_lr_mse[i]).iloc[:, 2:] for i in range(5)])/5
    df3 = sum([pd.DataFrame(robustness_dicts_naive[i]).iloc[:, 2:] for i in range(5)])/5  
    df4 = sum([pd.DataFrame(robustness_dicts_lr_mse[i]).iloc[:, 2:] for i in range(5)])/5
    df5 = sum([pd.DataFrame(robustness_dicts_interval_lr_mae[i]).iloc[:, 2:] for i in range(5)])/5
    df6 = sum([pd.DataFrame(robustness_dicts_lr_mae[i]).iloc[:, 2:] for i in range(5)])/5
    df7 = sum([pd.DataFrame(robustness_dicts_interval_rf_mse[i]).iloc[:, 2:] for i in range(5)])/5  
    df8 = sum([pd.DataFrame(robustness_dicts_rf_mse[i]).iloc[:, 2:] for i in range(5)])/5
    df9 = sum([pd.DataFrame(robustness_dicts_interval_rf_mae[i]).iloc[:, 2:] for i in range(5)])/5  
    df10 = sum([pd.DataFrame(robustness_dicts_rf_mae[i]).iloc[:, 2:] for i in range(5)])/5

    print("Converting fractions to percentages")
    heatmap_data1 = df1.multiply(100).values
    heatmap_data2 = df2.multiply(100).values
    heatmap_data3 = df3.multiply(100).values
    heatmap_data4 = df4.multiply(100).values
    heatmap_data5 = df5.multiply(100).values
    heatmap_data6 = df6.multiply(100).values
    heatmap_data7 = df7.multiply(100).values
    heatmap_data8 = df8.multiply(100).values
    heatmap_data9 = df9.multiply(100).values
    heatmap_data10 = df10.multiply(100).values

    # Labels
    x_labels = df1.columns.tolist()
    y_labels = ratios

    # Plot each heatmap
    heatmaps = []
    heatmaps.append(plot_heatmap(axes[0, 0], heatmap_data1, x_labels, y_labels, 'Meyer et al. (Naive Approach)'))
    heatmaps.append(plot_heatmap(axes[0, 1], heatmap_data3, x_labels, y_labels, 'ZORRO (Naive Approach)'))
    heatmaps.append(plot_heatmap(axes[1, 0], heatmap_data2, x_labels, y_labels, 'Meyer et al. (LinReg, mse)'))
    heatmaps.append(plot_heatmap(axes[1, 1], heatmap_data4, x_labels, y_labels, 'ZORRO (LinReg, mse)'))
    heatmaps.append(plot_heatmap(axes[2, 0], heatmap_data5, x_labels, y_labels, 'Meyer et al. (LinReg, mae)'))
    heatmaps.append(plot_heatmap(axes[2, 1], heatmap_data6, x_labels, y_labels, 'ZORRO (LinReg, mae)'))
    heatmaps.append(plot_heatmap(axes[3, 0], heatmap_data7, x_labels, y_labels, 'Meyer et al. (RndFrst, mse)'))
    heatmaps.append(plot_heatmap(axes[3, 1], heatmap_data8, x_labels, y_labels, 'ZORRO (RndFrst, mse)'))
    heatmaps.append(plot_heatmap(axes[4, 0], heatmap_data9, x_labels, y_labels, 'Meyer et al. (RndFrst, mae)'))
    heatmaps.append(plot_heatmap(axes[4, 1], heatmap_data10, x_labels, y_labels, 'ZORRO (RndFrst, mae)'))

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            print(f"Axes[{i}, {j}] images: {ax.images}")

    # Adjust layout and add colorbar
    plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.1, left=0.1, right=0.9)
    cb = fig.colorbar(heatmaps[-1], ax=axes.ravel().tolist(), orientation='vertical', pad=0.02)
    cb.set_label('Robustness Ratio (%)', fontsize=12)
    
    plt.savefig(f"{output_dir}/{args.dataset}-LeaveOneOut-method-heatmap.pdf", bbox_inches='tight')
  
    print("")
    print("Leave One Out finished!")
    print("")

# Main function to parse arguments
def main():
    # Argument parsing
    print("")
    print("Grabbing Arguments...")
    print("")
    parser = argparse.ArgumentParser(description="Run robustness tests")
    parser.add_argument('--task', choices=['Pattern_Mining', 'Pattern_Testing(Flawed)', 'Pattern_Testing(Set_Percent)', 'Normalization', 'leave_one_out'], help="Task to run on terminal: 'Pattern_Mining', 'Pattern_Testing(Flawed)', 'Testing', 'Normalization', 'leave_one_out'")
    parser.add_argument("--dataset", choices=["mpg", "ins", "bos", "fire"], help="Name of dataset utilized: 'mpg', 'ins', 'bos', 'fire'")
    args = parser.parse_args()

    # Set parameters
    # set parameters
    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.task == "leave_one_out":
        if args.dataset == "mpg":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 2, maximize=False)
        elif args.dataset == "ins":
            ratios = [0.02, 0.04, 0.06, 0.08]
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 500, maximize=False)
        elif args.dataset == "bos":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_Boston_cleaned(random_seed=params["random_seed"])
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 500, maximize=False)
        elif args.dataset == "fire":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 500, maximize=False)
        else:
            print("")
            print("Dataset is not provided, please provided dataset.")
    elif args.test == "discretization":
        if args.dataset == "mpg":
            #rndFrst_mae, rndFrst_mse, linReg_mae, linReg_mse
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            bin_numbers = [44, 41, 44, 23]
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            run_discretization_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 2, bin_numbers, max_uncertain_pct=10, maximize=True)
        elif args.dataset == "ins":
            ratios = [0.02, 0.04, 0.06, 0.08]
            bin_numbers = [41, 38, 15, 36]
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            run_discretization_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 500, bin_numbers, max_uncertain_pct=10, maximize=True)
        else:
            print("")
            print("Dataset is not provided, please provided dataset.")
    else:
            print("")
            print("No task method given. Please declare type of task to run")


if __name__ == "__main__":
  main()


