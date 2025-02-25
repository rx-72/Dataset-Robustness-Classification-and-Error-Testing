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
from itertools import combinations, product
import heapq

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

with open("data-params.json", "r") as f:
  params = json.load(f)

METRICS = {
    "accuracy": accuracy
}
def pattern_mining(X_train, X_test, y_train, y_test, column_2_bins, num_bins, pattern_size, rr, ur, thresholds=None, args=None):
  X_train_transformed = X_train.copy()
  X_test_transformed = X_test.copy()

  columns_to_bin = column_2_bins
  n_bins = num_bins
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins
  def compute_optimized_candidates(X, tau, n_features_to_combine=2):

    candidates = []
    n_samples, n_features = X.shape
    
    # Generate combinations of features to combine
    feature_combinations = list(combinations(range(n_features), n_features_to_combine))
    
    for feature_combo in tqdm(feature_combinations, desc="Processing feature combinations"):
        # Get unique values for each feature in the combination
        values_list = [np.unique(X.iloc[:, idx]) for idx in feature_combo]
        
        # Iterate over all possible value-condition combinations
        for value_combo in tqdm(
            product(*values_list),
            desc=f"Processing combination {feature_combo}",
            leave=False
        ):
            for conditions in product(["<", "=", ">"], repeat=n_features_to_combine):
                # Skip conflicting combinations
                if has_conflicts(value_combo, conditions):
                    continue

                # Generate pattern
                pattern = np.ones(n_samples, dtype=bool)  # Start with all True
                for idx, (feat_idx, val, cond) in enumerate(zip(feature_combo, value_combo, conditions)):
                    if cond == "<":
                        pattern &= X.iloc[:, feat_idx] < val
                    elif cond == "=":
                        pattern &= X.iloc[:, feat_idx] == val
                    elif cond == ">":
                        pattern &= X.iloc[:, feat_idx] > val
                
                # Check support
                support = np.sum(pattern)
                if support <= (tau * n_samples): #and support >= (0.5 * tau * n_samples):
                    candidates.append((feature_combo, value_combo, conditions))
    
    return candidates


  def has_conflicts(value_combo, conditions):
    for i in range(len(value_combo)):
        for j in range(i + 1, len(value_combo)):
            # Detect conflicts like "< x AND > x" for the same value
            if conditions[i] == ">" and conditions[j] == "<" and value_combo[i] <= value_combo[j]:
                return True
            if conditions[i] == "<" and conditions[j] == ">" and value_combo[i] >= value_combo[j]:
                return True
            # Detect conflicts like "= x AND < x" or "= x AND > x"
            if conditions[i] == "=" and (conditions[j] == "<" and value_combo[i] >= value_combo[j]):
                return True
            if conditions[i] == "=" and (conditions[j] == ">" and value_combo[i] <= value_combo[j]):
                return True
            # Symmetry of conditions
            if conditions[j] == "=" and (conditions[i] == "<" and value_combo[j] >= value_combo[i]):
                return True
            if conditions[j] == "=" and (conditions[i] == ">" and value_combo[j] <= value_combo[i]):
                return True
    return False

  print("")
  print("Generating Candidate Patterns for dataset...")
  print("")

  candidates = compute_optimized_candidates(X_train_transformed, 0.1, n_features_to_combine=pattern_size)
  
  def get_complex_candidate_target_indices(df, candidate):
    columns, values, conditions = candidate
    mask = pd.Series(True, index=df.index)  # Start with a mask that selects all rows

    for col, val, cond in zip(columns, values, conditions):
        if cond == "<":
            mask &= df.iloc[:, col] < val
        elif cond == "=":
            mask &= df.iloc[:, col] == val
        elif cond == ">":
            mask &= df.iloc[:, col] > val
        else:
            raise ValueError(f"Unsupported condition: {cond}")

    return df[mask].index.tolist()
  
  def filter_dataframe_by_complex_candidate(df, labels, candidate):
    mask = get_complex_candidate_target_indices(df, candidate)

    return df[df.index.isin(mask) == False], labels[labels.index.isin(mask) == False]

  def mse(y_true, y_pred):
    return sum((y_true - y_pred)**2)/len(y_true)

  def mae(y_true, y_pred):
    return sum(abs(y_true - y_pred))/len(y_true)

  lr = LinearRegression()

  def mse_filter(candidates, X_train_transformed, X_test_transformed, y_train, y_test):

    top_candidates = []

    seen_mse = set()
    
    for candidate in tqdm(candidates, desc="Processing Candidates"):
        number_indices = len(get_complex_candidate_target_indices(X_train_transformed, candidate))
        filtered_X_train, filtered_y_train = filter_dataframe_by_complex_candidate(X_train_transformed, y_train, candidate)
        model = lr.fit(filtered_X_train, filtered_y_train)
        predictions = model.predict(X_test_transformed)
        new_mse = mse(predictions, y_test)
        #print(new_mae)

        if new_mse in seen_mse:
            continue
        
        seen_mse.add(new_mse)

        top_candidates = top_candidates + [(new_mse, candidate, number_indices)]
    if thresholds == None:
      top_candidates = sorted(top_candidates, key=lambda x: (x[0], -x[2]), reverse=True)
    else:
      top_candidates = sorted(top_candidates, key=lambda x: (x[0], -x[2]), reverse=True)[thresholds[0]:thresholds[1]]  
    
    
    return top_candidates

  print("")
  print("Filtering Candidates...")
  print("")
  
  top_candidates = [x[1] for x in mse_filter(candidates, X_train_transformed, X_test_transformed, y_train, y_test)]

  if args != None:
    top_candidates[500] = ((9, 12), (398.0, 6), ('>', '<'))
  
  print("")
  print(f"Remaining Number of Top Candidates found: {len(top_candidates)}")
  print("")

  def top_k_finder(candidates, X_transformed, X_train, y_train, X_test, y_test, k, tau, robustness_radius, uncertain_ratio):
    n_samples, n_features = X_train.shape

    seen_ratios = set()
    
    # Min-heap to store the top k worst robustness ratios
    top_k_patterns = []
    
    for candidate in tqdm(candidates, desc="Processing Candidates"):
        # Get indices satisfying the candidate pattern
        target_indices = get_complex_candidate_target_indices(X_transformed, candidate)
        
        # Compute robustness ratio
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(
            X_train, y_train, X_test, y_test, 
            uncertain_num=int(tau * len(y_train)),
            boundary_indices=target_indices,
            uncertain_radius=uncertain_ratio * (y_train.max() - y_train.min()), 
            robustness_radius=robustness_radius, 
            interval=False
        )

        if robustness_ratio in seen_ratios:
            continue

        seen_ratios.add(robustness_ratio)
        
        #print(robustness_ratio)
        
        top_k_patterns = top_k_patterns + [(robustness_ratio, candidate)]
        
    return sorted(top_k_patterns, key=lambda x: x[0])[0:k]

  print("")
  print(f"Organizing results for top 3 patterns based on Worst Error injection Results...")
  print("")

  result = top_k_finder(top_candidates, X_train_transformed, X_train, y_train, X_test, y_test, 10, 0.1, rr, ur)

  print("")
  print(result)
  print("")
  
  lst = []
  threshold_count = len(X_train)*0.1*0.5
  print(f"Threshold: {threshold_count}")
  print("")
  for i in result:
    def matchcounter(x, y):
        count = 0
    
        for i in x:
            if i in y:
                count += 1
        return count
    if len(lst) == 0:
        lst = [i]
    else:
        test = 0
        for j in lst:
            indices_pattern1 = get_complex_candidate_target_indices(X_train_transformed, i[1])
            indices_pattern2 = get_complex_candidate_target_indices(X_train_transformed, j[1])
            #print(matchcounter(indices_pattern1, indices_pattern2))
            if matchcounter(indices_pattern1, indices_pattern2) < threshold_count:
                test += 1
                if test == len(lst):
                    lst = lst + [i]
            else:
                break

  print(lst)
  
  pattern1 = lst[0][1]
  pattern2 = lst[1][1]
  pattern3 = lst[2][1]
  robustness1 = lst[0][0]
  robustness2 = lst[1][0]
  robustness3 = lst[2][0]

  target_indices_of_pattern1 = get_complex_candidate_target_indices(X_train_transformed, pattern1)
  target_indices_of_pattern2 = get_complex_candidate_target_indices(X_train_transformed, pattern2)
  target_indices_of_pattern3 = get_complex_candidate_target_indices(X_train_transformed, pattern3)

  print("")
  print("The top 3 patterns found are as follows:")
  print("")
  
  cols = X_train.columns
  print("Pattern 1:")
  for i in range(0, len(pattern1[0])):
    target_column = cols[pattern1[0][i]]
    val = pattern1[1][i]
    condition = pattern1[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print("")
  print(f"Robustness results after utilizing pattern 1: {robustness1}")
  print("")
  print("")
  print("")
  print("Pattern 2:")
  for i in range(0, len(pattern2[0])):
    target_column = cols[pattern2[0][i]]
    val = pattern2[1][i]
    condition = pattern2[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print("")
  print(f"Robustness results after utilizing pattern 2: {robustness2}")
  print("")
  print("")
  print("")
  print("Pattern 3:")
  for i in range(0, len(pattern3[0])):
    target_column = cols[pattern3[0][i]]
    val = pattern3[1][i]
    condition = pattern3[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print("")
  print(f"Robustness results after utilizing pattern 3: {robustness3}")
  print("")
  
def pattern_testing_line(X_train, X_test, y_train, y_test, column_2_bins, num_bins, ratios, p1, p2, p3, rr, output_dir, args, thresholds):
  X_train_transformed = X_train.copy()
  X_test_transformed = X_test.copy()

  columns_to_bin = column_2_bins
  n_bins = num_bins
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  def get_complex_candidate_target_indices(df, candidate):
    columns, values, conditions = candidate
    mask = pd.Series(True, index=df.index)  # Start with a mask that selects all rows

    for col, val, cond in zip(columns, values, conditions):
        if cond == "<":
            mask &= df.iloc[:, col] < val
        elif cond == "=":
            mask &= df.iloc[:, col] == val
        elif cond == ">":
            mask &= df.iloc[:, col] > val
        else:
            raise ValueError(f"Unsupported condition: {cond}")

    return df[mask].index.tolist()

  pattern1 = p1
  pattern2 = p2
  pattern3 = p3

  target_indices_of_pattern1 = get_complex_candidate_target_indices(X_train_transformed, pattern1)
  target_indices_of_pattern2 = get_complex_candidate_target_indices(X_train_transformed, pattern2)
  target_indices_of_pattern3 = get_complex_candidate_target_indices(X_train_transformed, pattern3)

  cols = X_train.columns
  print("")
  print("Pattern 1:")
  for i in range(0, len(pattern1[0])):
    target_column = cols[pattern1[0][i]]
    val = pattern1[1][i]
    condition = pattern1[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print()
  print("Pattern 2:")
  for i in range(0, len(pattern2[0])):
    target_column = cols[pattern2[0][i]]
    val = pattern2[1][i]
    condition = pattern2[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print()
  print("Pattern 3:")
  for i in range(0, len(pattern3[0])):
    target_column = cols[pattern3[0][i]]
    val = pattern3[1][i]
    condition = pattern3[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print("")
  
  print("")
  print("pattern1_testing (Meyer)")
  print("")
  robustness_ratio_range_pattern1_meyer = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern1,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
        robustness_ratio_range_pattern1_meyer = robustness_ratio_range_pattern1_meyer + [robustness_ratio]

  print("")
  print("pattern1_testing (Zorro)")
  print("")
  robustness_ratio_range_pattern1_zorro = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
        
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern1,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
        robustness_ratio_range_pattern1_zorro = robustness_ratio_range_pattern1_zorro + [robustness_ratio]

  print("")
  print("pattern2_testing (Meyer)")
  print("")
  robustness_ratio_range_pattern2_meyer = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern2,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
        robustness_ratio_range_pattern2_meyer = robustness_ratio_range_pattern2_meyer + [robustness_ratio]

  print("")
  print("pattern2_testing (Zorro)")
  print("")
  robustness_ratio_range_pattern2_zorro = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern2,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
        robustness_ratio_range_pattern2_zorro = robustness_ratio_range_pattern2_zorro + [robustness_ratio]

  print("")  
  print("pattern3_testing (Meyer)")
  print("")
  robustness_ratio_range_pattern3_meyer = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern3,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
        robustness_ratio_range_pattern3_meyer = robustness_ratio_range_pattern3_meyer + [robustness_ratio]

  print("") 
  print("pattern3_testing (Zorro)")
  print("") 
  robustness_ratio_range_pattern3_zorro = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern3,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
        robustness_ratio_range_pattern3_zorro = robustness_ratio_range_pattern3_zorro + [robustness_ratio]

  print("") 
  print("Naive (Zorro)")
  print("") 
  robustness_naive = np.zeros(len(ratios))
  for seed in tqdm(range(5), desc=f'Progress'):
    placeholder = []
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
        robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
        placeholder = placeholder + [robustness_ratio]
    robustness_naive = robustness_naive + np.array(placeholder)
  robustness_naive = (robustness_naive/5).tolist()

  print("") 
  print("Naive (Meyer)")
  print("") 
  
  robustness_naive_interval = np.zeros(len(ratios))
  for seed in tqdm(range(5), desc=f'Progress'):
    placeholder = []
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())

    uncertain_pct = 0.1
    uncertain_num = int(uncertain_pct*len(y_train))
    for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
        robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
        placeholder = placeholder + [robustness_ratio]
    robustness_naive_interval = robustness_naive_interval + np.array(placeholder)
  robustness_naive_interval = (robustness_naive_interval/5).tolist()

  def average_over_multiple_large_robustness_loss(robustness_lst, x):
    sum_vals = 0
    count = 0
    previous = -1
    for i in range(0, len(robustness_lst)):
        if ratios[i] > x:
            break
        sum_vals += robustness_lst[i]
        count += 1
    average = sum_vals/count
    statement = f"Average Robustness was {average} over {count} ratio recordings ({x} threshold)" #with {count - 1} large decreasing steps"
    return statement

  print("")
  print("Pattern 1 Average:")
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern1_zorro, thresholds[0]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern1_zorro, thresholds[1]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern1_zorro, thresholds[2]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern1_zorro, thresholds[3]))
  print("")
  print("Pattern 2 Average:")
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern2_zorro, thresholds[0]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern2_zorro, thresholds[1]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern2_zorro, thresholds[2]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern2_zorro, thresholds[3]))
  print("")
  print("Pattern 3 Average:")
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern3_zorro, thresholds[0]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern3_zorro, thresholds[1]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern3_zorro, thresholds[2]))
  print(average_over_multiple_large_robustness_loss(robustness_ratio_range_pattern3_zorro, thresholds[3]))
  print("")

  # Create the line plots with a 4x2 grid
  fig, axes = plt.subplots(4, 2, figsize=(15, 18), dpi=200)

  # Function to plot a single line plot with annotations
  def plot_line(ax, x_values, y_values, title):
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='orange', alpha=0.8, label='_nolegend_')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Uncertainty Radius (%)', fontsize=12)
    ax.set_ylabel('Robustness Ratio (%)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set(ylim = (-0.05, 1.2))
    
    # Add text annotations for each point with dynamic offset
    #y_offset = (max(y_values) - min(y_values)) * 0.05  # 5% of the y-range as offset
    #for x, y in zip(x_values, y_values):
    #    if y > 0:
    #        ax.text(x, y + y_offset, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
    #    else:
    #        ax.text(x, y + y_offset * 2, f'{y:.3f}', ha='center', va='bottom', fontsize=8, color='gray')
    
    #ax.legend()

  # Plot each line
  plot_line(axes[0, 0], ratios, robustness_naive_interval, 'Meyer et al. (Naive Approach)')
  plot_line(axes[0, 1], ratios, robustness_naive, 'ZORRO (Naive Approach)')
  plot_line(axes[1, 0], ratios, robustness_ratio_range_pattern1_meyer, 'Meyer et al. (Pattern 1)')
  plot_line(axes[1, 1], ratios, robustness_ratio_range_pattern1_zorro, 'ZORRO (Pattern 1)')
  plot_line(axes[2, 0], ratios, robustness_ratio_range_pattern2_meyer, 'Meyer et al. (Pattern 2)')
  plot_line(axes[2, 1], ratios, robustness_ratio_range_pattern2_zorro, 'ZORRO (Pattern 2)')
  plot_line(axes[3, 0], ratios, robustness_ratio_range_pattern3_meyer, 'Meyer et al. (Pattern 3)')
  plot_line(axes[3, 1], ratios, robustness_ratio_range_pattern3_zorro, 'ZORRO (Pattern 3)')

  # Adjust layout
  plt.subplots_adjust(wspace=0.3, hspace=0.6, bottom=0.1, left=0.1, right=0.9)

  # Save the figure
  plt.savefig(f'{output_dir}/Pattern_Robustness_Decreasing(Meyer and Zorro)_{args.dataset}.pdf', bbox_inches='tight')

  # Create the line plots with a 4x2 grid
  fig, axes = plt.subplots(4, 1, figsize=(15, 18), dpi=200)
  # Function to plot a single line plot with annotations
  def plot_line(ax, x_values, y_values, title):
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='orange', alpha=0.8, label='_nolegend_')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Uncertainty Radius (%)', fontsize=12)
    ax.set_ylabel('Robustness Ratio (%)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set(ylim = (-0.05, 1.2))
    
    # Add text annotations for each point with dynamic offset
    y_offset = (max(y_values) - min(y_values)) * 0.05  # 5% of the y-range as offset
    for x, y in zip(x_values, y_values):
        if y > 0:
            ax.text(x, y + y_offset, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(x, y + y_offset * 2, f'{y:.3f}', ha='center', va='bottom', fontsize=8, color='gray')
    
    ax.legend()

  # Plot each line
  plot_line(axes[0], ratios, robustness_naive, 'ZORRO (Naive Approach)')
  plot_line(axes[1], ratios, robustness_ratio_range_pattern1_zorro, 'ZORRO (Pattern 1)')
  plot_line(axes[2], ratios, robustness_ratio_range_pattern2_zorro, 'ZORRO (Pattern 2)')
  plot_line(axes[3], ratios, robustness_ratio_range_pattern3_zorro, 'ZORRO (Pattern 3)')

  # Adjust layout
  plt.subplots_adjust(wspace=0.3, hspace=0.6, bottom=0.1, left=0.1, right=0.9)

  # Save the figure
  plt.savefig(f'{output_dir}/Pattern_Robustness_Decreasing(Zorro Only, Annontations)_{args.dataset}.pdf', bbox_inches='tight')

  plt.figure(figsize=(30, 30))
  plt.xlim(0, 0.5)
  colors = ['blue', 'red', 'green', 'purple']
  linestyles = ['-', '-', '-', '-']
  labels = ['Naive', 'Pattern 1', 'Pattern 2', 'Pattern 3']
  patterns = [
    robustness_naive, 
    robustness_ratio_range_pattern1_zorro, 
    robustness_ratio_range_pattern2_zorro, 
    robustness_ratio_range_pattern3_zorro
  ]
  for i, (pattern, color, linestyle, label) in enumerate(zip(patterns, colors, linestyles, labels)):
    pattern = np.array(pattern)  # Convert list to NumPy array
    plt.plot(ratios, pattern, color=color, linestyle=linestyle, linewidth=3, label=label)
    
  for y in [0.8, 0.6, 0.2]:
    plt.axhline(y=y, linestyle='dotted', color='black', linewidth=2)
  
  plt.xlabel('Uncertainty Radius (%)', fontsize=24)
  plt.ylabel('Robustness Ratio (%)', fontsize=24)
  plt.xticks(fontsize=24)
  plt.yticks(fontsize=24)
  plt.legend(fontsize=30, loc='lower left')
  plt.savefig(f'{output_dir}/Multi_Robustness_lineplot_(Zorro w\ annotations)_{args.dataset}.pdf', bbox_inches='tight')

  print("")
  print("Graphing Completed")
  print("")


def pattern_testing_heat(X_train, X_test, y_train, y_test, column_2_bins, num_bins, ratios, p1, p2, p3, rr, output_dir, args):
  X_train_transformed = X_train.copy()
  X_test_transformed = X_test.copy()

  columns_to_bin = column_2_bins
  n_bins = num_bins
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  def get_complex_candidate_target_indices(df, candidate):
    columns, values, conditions = candidate
    mask = pd.Series(True, index=df.index)  # Start with a mask that selects all rows

    for col, val, cond in zip(columns, values, conditions):
        if cond == "<":
            mask &= df.iloc[:, col] < val
        elif cond == "=":
            mask &= df.iloc[:, col] == val
        elif cond == ">":
            mask &= df.iloc[:, col] > val
        else:
            raise ValueError(f"Unsupported condition: {cond}")

    return df[mask].index.tolist()

  pattern1 = p1
  pattern2 = p2
  pattern3 = p3

  target_indices_of_pattern1 = get_complex_candidate_target_indices(X_train_transformed, pattern1)
  target_indices_of_pattern2 = get_complex_candidate_target_indices(X_train_transformed, pattern2)
  target_indices_of_pattern3 = get_complex_candidate_target_indices(X_train_transformed, pattern3)

  cols = X_train.columns
  print("")
  print("Pattern 1:")
  for i in range(0, len(pattern1[0])):
    target_column = cols[pattern1[0][i]]
    val = pattern1[1][i]
    condition = pattern1[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print()
  print("Pattern 2:")
  for i in range(0, len(pattern2[0])):
    target_column = cols[pattern2[0][i]]
    val = pattern2[1][i]
    condition = pattern2[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print()
  print("Pattern 3:")
  for i in range(0, len(pattern3[0])):
    target_column = cols[pattern3[0][i]]
    val = pattern3[1][i]
    condition = pattern3[2][i]
    if target_column in column_bins:
        binnings = column_bins[target_column]
        if condition == '>':
            val = binnings[val]
            print(f"'{target_column}' {condition} {val}")
        elif condition == '<':
            val = binnings[val - 1]
            print(f"'{target_column}' {condition} {val}")
        else:
            lower_val = binnings[val - 1]
            upper_val = binnings[val]
            print(f"{lower_val} < '{target_column}' < {upper_val}")
    else:
        print(f"'{target_column}' {condition} {val}")
  print("")

  print("pattern1_testing (Meyer)")
  robustness_dicts_interval_pattern1 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
    robustness_dict_interval = dict()
    robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
    robustness_dict_interval['uncertain_radius_ratios'] = ratios
    for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
        robustness_dict_interval[uncertain_pct] = list()
        uncertain_num = int(uncertain_pct*len(y_train))
        for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
            robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern1,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
            robustness_dict_interval[uncertain_pct].append(robustness_ratio)
    robustness_dicts_interval_pattern1.append(robustness_dict_interval)

  print("pattern1_testing (Zorro)")
  robustness_dicts_pattern1 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
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
                                                                    boundary_indices=target_indices_of_pattern1,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
            robustness_dict[uncertain_pct].append(robustness_ratio)
    robustness_dicts_pattern1.append(robustness_dict)

  print("pattern2_testing (Meyer)")
  robustness_dicts_interval_pattern2 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
    robustness_dict_interval = dict()
    robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
    robustness_dict_interval['uncertain_radius_ratios'] = ratios
    for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
        robustness_dict_interval[uncertain_pct] = list()
        uncertain_num = int(uncertain_pct*len(y_train))
        for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
            robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern2,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
            robustness_dict_interval[uncertain_pct].append(robustness_ratio)
    robustness_dicts_interval_pattern2.append(robustness_dict_interval)

  
  print("pattern2_testing (Zorro)")
  robustness_dicts_pattern2 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
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
                                                                    boundary_indices=target_indices_of_pattern2,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
            robustness_dict[uncertain_pct].append(robustness_ratio)
    robustness_dicts_pattern2.append(robustness_dict)

  print("pattern3_testing (Meyer)")
  robustness_dicts_interval_pattern3 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
    robustness_dict_interval = dict()
    robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
    robustness_dict_interval['uncertain_radius_ratios'] = ratios
    for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
        robustness_dict_interval[uncertain_pct] = list()
        uncertain_num = int(uncertain_pct*len(y_train))
        for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
            robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=target_indices_of_pattern3,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
            robustness_dict_interval[uncertain_pct].append(robustness_ratio)
    robustness_dicts_interval_pattern3.append(robustness_dict_interval)

  print("pattern3_testing (Zorro)")
  robustness_dicts_pattern3 = []
  for seed in range(1):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
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
                                                                    boundary_indices=target_indices_of_pattern3,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
            robustness_dict[uncertain_pct].append(robustness_ratio)
    robustness_dicts_pattern3.append(robustness_dict)

  print("Naive tests")
  robustness_dicts_naive = []
  for seed in tqdm(range(5), desc=f'Progress'):
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
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

  print("Naive tests (meyer)")
  robustness_dicts_interval_naive = []
  for seed in tqdm(range(5), desc=f'Progress'):
    # mpg +- 2 is robust
    robustness_radius = rr
    label_range = (y_train.max()-y_train.min())
    ratios = ratios
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
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

  # Create the heatmap plot with a 2x2 grid
  fig, axes = plt.subplots(4, 2, figsize=(15, 18), dpi=200)

  # Define colormap
  cmap = plt.get_cmap("autumn_r")

  # Function to plot a single heatmap
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

  # Data for the heatmaps

  #naive
  df1 = sum([pd.DataFrame(robustness_dicts_interval_naive[i]).iloc[:, 2:] for i in range(5)])/5
  df2 = sum([pd.DataFrame(robustness_dicts_naive[i]).iloc[:, 2:] for i in range(5)])/5

  #pattern1
  df3 = sum([pd.DataFrame(robustness_dicts_interval_pattern1[i]).iloc[:, 2:] for i in range(1)])/1  
  df4 = sum([pd.DataFrame(robustness_dicts_pattern1[i]).iloc[:, 2:] for i in range(1)])/1

  #pattern2
  df5 = sum([pd.DataFrame(robustness_dicts_interval_pattern2[i]).iloc[:, 2:] for i in range(1)])/1
  df6 = sum([pd.DataFrame(robustness_dicts_pattern2[i]).iloc[:, 2:] for i in range(1)])/1

  #pattern3
  df7 = sum([pd.DataFrame(robustness_dicts_interval_pattern3[i]).iloc[:, 2:] for i in range(1)])/1
  df8 = sum([pd.DataFrame(robustness_dicts_pattern3[i]).iloc[:, 2:] for i in range(1)])/1


  # Convert fractions to percentages
  heatmap_data1 = df1.multiply(100).values
  heatmap_data2 = df2.multiply(100).values
  heatmap_data3 = df3.multiply(100).values
  heatmap_data4 = df4.multiply(100).values
  heatmap_data5 = df5.multiply(100).values
  heatmap_data6 = df6.multiply(100).values
  heatmap_data7 = df7.multiply(100).values
  heatmap_data8 = df8.multiply(100).values


  # Labels
  x_labels = df1.columns.tolist()
  y_labels = ratios

  # Plot each heatmap
  plot_heatmap(axes[0, 0], heatmap_data1, x_labels, y_labels, 'Meyer et al. (Naive Approach)')
  plot_heatmap(axes[0, 1], heatmap_data2, x_labels, y_labels, 'ZORRO (Naive Approach)')
  plot_heatmap(axes[1, 0], heatmap_data3, x_labels, y_labels, 'Meyer et al. (Pattern 1)')
  plot_heatmap(axes[1, 1], heatmap_data4, x_labels, y_labels, 'ZORRO (Pattern 1)')
  plot_heatmap(axes[2, 0], heatmap_data5, x_labels, y_labels, 'Meyer et al. (Pattern 2)')
  plot_heatmap(axes[2, 1], heatmap_data6, x_labels, y_labels, 'ZORRO (Pattern 2)')
  plot_heatmap(axes[3, 0], heatmap_data7, x_labels, y_labels, 'Meyer et al. (Pattern 3)')
  plot_heatmap(axes[3, 1], heatmap_data8, x_labels, y_labels, 'ZORRO (Pattern 3)')


  # Adjust layout and add colorbar
  plt.subplots_adjust(wspace=0.2, hspace=0.6, bottom=0.1, left=0.1, right=0.9)
  cb = fig.colorbar(axes[0, 1].images[0], ax=axes, orientation='vertical', pad=0.02)
  cb.set_label('Robustness Ratio (%)', fontsize=12)
  plt.savefig(f'{output_dir}/Pattern_testing_{args.dataset}.pdf', bbox_inches='tight')

  print("Graph has been created")

def normalization_all(X_train_ins, X_test_ins, y_train_ins, y_test_ins, X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg, X_train_bos, X_test_bos, y_train_bos, y_test_bos, X_train_fire, X_test_fire, y_train_fire, y_test_fire):
  print("Getting things started...")
  X_train_transformed = X_train_ins.copy()
  X_test_transformed = X_test_ins.copy()

  columns_to_bin = ['bmi']
  n_bins = {'bmi': int(np.sqrt(496))}
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train_ins[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test_ins[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  def get_complex_candidate_target_indices(df, candidate):
    columns, values, conditions = candidate
    mask = pd.Series(True, index=df.index)  # Start with a mask that selects all rows

    for col, val, cond in zip(columns, values, conditions):
      if cond == "<":
        mask &= df.iloc[:, col] < val
      elif cond == "=":
        mask &= df.iloc[:, col] == val
      elif cond == ">":
        mask &= df.iloc[:, col] > val
      else:
        raise ValueError(f"Unsupported condition: {cond}")
    return df[mask].index.tolist()

  pattern_ins = ((1,), (4,), ('<',))
  target_indices_of_pattern_ins = get_complex_candidate_target_indices(X_train_transformed, pattern_ins)
  boundary_indices_lst = [target_indices_of_pattern_ins]

  X_train_transformed = X_train_mpg.copy()
  X_test_transformed = X_test_mpg.copy()

  columns_to_bin = ['displacement', 'horsepower', 'weight', 'acceleration']
  n_bins = {'displacement': int(np.sqrt(72)), 'horsepower': int(np.sqrt(87)), 'weight': int(np.sqrt(288)), 'acceleration': int(np.sqrt(86))}
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train_mpg[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test_mpg[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  pattern_mpg =  ((3, 4, 5), (9, 3, 82.0), ('<', '<', '<'))
  target_indices_of_pattern_mpg = get_complex_candidate_target_indices(X_train_transformed, pattern_mpg)
  boundary_indices_lst = boundary_indices_lst + [target_indices_of_pattern_mpg]

  X_train_transformed = X_train_bos.copy()
  X_test_transformed = X_test_bos.copy()

  columns_to_bin = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']
  n_bins = {'CRIM': int(np.sqrt(402)), 'INDUS': int(np.sqrt(72)), 'NOX': int(np.sqrt(76)), 'RM': int(np.sqrt(366)), 'AGE': int(np.sqrt(302)), 'DIS': int(np.sqrt(339)), 'B': int(np.sqrt(287)), 'LSTAT': int(np.sqrt(366))}
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train_bos[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test_bos[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  pattern_bos = ((5, 10), (10, 17.9), ('>', '>'))
  target_indices_of_pattern_bos = get_complex_candidate_target_indices(X_train_transformed, pattern_bos)
  boundary_indices_lst = boundary_indices_lst + [target_indices_of_pattern_bos]

  X_train_transformed = X_train_fire.copy()
  X_test_transformed = X_test_fire.copy()

  columns_to_bin = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
  n_bins = {'FFMC': int(np.sqrt(103)), 'DMC': int(np.sqrt(199)), 'DC': int(np.sqrt(199)), 'ISI': int(np.sqrt(112)), 'temp': int(np.sqrt(183)), 'RH': int(np.sqrt(73))}
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train_fire[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test_fire[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  pattern_fire = ((6, 7), (5, 6), ('=', '<'))
  target_indices_of_pattern_fire = get_complex_candidate_target_indices(X_train_transformed, pattern_fire)
  boundary_indices_lst = boundary_indices_lst + [target_indices_of_pattern_fire]

  uncertain_radius_ins = 0.25*(y_train_ins.max() - y_train_ins.min())
  uncertain_radius_mpg = 0.25*(y_train_mpg.max() - y_train_mpg.min())
  uncertain_radius_bos = 0.25*(y_train_bos.max() - y_train_bos.min())
  uncertain_radius_fire = 0.25*(y_train_fire.max() - y_train_fire.min())
  uncertain_radii = [uncertain_radius_ins, uncertain_radius_mpg, uncertain_radius_bos, uncertain_radius_fire]

  uncertain_percentage = 0.1
  uncertain_num_ins = int(uncertain_percentage*len(y_train_ins))
  uncertain_num_mpg = int(uncertain_percentage*len(y_train_mpg))
  uncertain_num_bos = int(uncertain_percentage*len(y_train_bos))
  uncertain_num_fire = int(uncertain_percentage*len(y_train_fire))
  uncertain_numbers = [uncertain_num_ins, uncertain_num_mpg, uncertain_num_bos, uncertain_num_fire]

  dataset_sizes = [len(y_train_ins), len(y_train_mpg), len(y_train_bos), len(y_train_fire)]
  dataset_names = ["Insurance", "MPG", "BOS", "FIRE"]

  dataset_dct = {}
  dataset_dct["Insurance"] = [X_train_ins, X_test_ins, y_train_ins, y_test_ins]
  dataset_dct["MPG"] = [X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg]
  dataset_dct["BOS"] = [X_train_bos, X_test_bos, y_train_bos, y_test_bos]
  dataset_dct["FIRE"] = [X_train_fire, X_test_fire, y_train_fire, y_test_fire]

  def robustness_score_normalization(uncertain_numbers, uncertain_radii, dataset_sizes, boundary_indices_lst, dataset_names, dataset_dct):
      robustness_radii_10 = [] #find robustness radius that grants radii robustness ratio of 0.10 or more 
                             #(alt. use 0.5 instead {depending on closeness, this ratio may need to be larger})

      for i in range(0, len(dataset_names)):
        uncertain_number = uncertain_numbers[i]
        uncertain_radius = uncertain_radii[i]
        boundary_indices = boundary_indices_lst[i]
        X_train, X_test, y_train, y_test = dataset_dct[dataset_names[i]]

        #print("target:")
        robustness_radius= 1
        if dataset_names[i] == "Insurance":
            radius_increment = 500
        elif dataset_names[i] == "FIRE":
            radius_increment = 1
        else:
            radius_increment = 0.01

        robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_number,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius,
                                                                    interval=False)

        print("Calculating best radius for " + dataset_names[i])

        #print(robustness_ratio)
        with tqdm(total=500, desc=f"Finding radius for {dataset_names[i]}", leave=False) as pbar:
            while robustness_ratio < 0.25:
                robustness_radius += radius_increment
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_number,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius,
                                                                    interval=False)
                pbar.update(radius_increment)
                #print(robustness_ratio)
        print(robustness_radius)
        #robustness_radii_10.append(robustness_radius)
        robustness_radii_10.append(np.log(1 + robustness_radius)/np.log(1 + (max(y_test) - min(y_test))))

      results = {}

      mean_radius = np.mean(robustness_radii_10)
      std_radius = np.std(robustness_radii_10)
    
      max_radii = max(robustness_radii_10)
      min_radii = min(robustness_radii_10)
      for i, dataset_name in enumerate(dataset_names):
        normalized_radius = 1 - ((robustness_radii_10[i] - mean_radius) / (std_radius + 1e-8))  # Prevent division by zero
        #normalized_size = (dataset_sizes[i]/max(dataset_sizes))
        #robustness_score = 0.5*normalized_radius + 0.5*normalized_size    
        robustness_score = normalized_radius
        #print(f"Normalized robustness score for {dataset_name} dataset is {robustness_score:.4f}")
        results[dataset_name] = robustness_score

      items = list(sorted(results.items(), key=lambda x: x[1]))
        
      for item in items:
        print(f"Normalized robustness score for {item[0]} dataset under worst case scenario is {item[1]:.4f}")

      worst = items[0]
      best = items[-1]
    
      print("")
      print("Thus we know the following:")
      print(f"The least robust dataset w/ repsect to assigned task is {worst[0]} with a normalized robustness score of {worst[1]:.4f}")
      print(f"The most robust dataset w/ repsect to assigned task is {best[0]} with a normalized robustness score of {best[1]:.4f}")
    
      return results
  result = robustness_score_normalization(uncertain_numbers, uncertain_radii, dataset_sizes, boundary_indices_lst, dataset_names, dataset_dct)

def normalization(X_train, y_train, X_test, y_test, bins, name, r_radius, r_radius_increment, columns_2_bin, number_bins, p1, p2, p3):

  print("Getting things started...")
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

  best_feature = select_best_feature(X_train, y_train)
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

  print("")
  print("Running tests on histogram method with Linear Regression, Mean Absolute Error...")
  
  tree = clf.tree_
  feature_names = X_train.columns
  best_thresholds = get_positive_paths(tree, feature_names)
  thresholds = [thres[best_feature] for thres in best_thresholds if best_feature in thres]
  boundary_indices_lr_mae, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bins[0])

  print("")
  print("Running tests on histogram method with Linear Regression, Mean Squared Error...")
  
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
  boundary_indices_lr_mse, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bins[1])

  print("")
  print("Running tests on histogram method with RandomForest Regressor, Mean Absolute Error...")

  rf = RandomForestRegressor
  X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
  boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, rf, mae, maximize=False)

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
  boundary_indices_rf_mae, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bins[2])

  print("")
  print("Running tests on histogram method with RandomForest Regressor, Mean Squared Error...")
  
  rf = RandomForestRegressor
  X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)  
  boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, rf, mse, maximize=False)

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
  boundary_indices_rf_mse, bin_edges = discretize_and_sample(X_train, best_feature, thresholds, total_samples=int(0.1 * len(X_train)), num_bins=bins[3])

  X_train_transformed = X_train.copy()
  X_test_transformed = X_test.copy()

  columns_to_bin = columns_2_bin
  n_bins = number_bins
  column_bins = {}

  for col in columns_to_bin:
    # Apply binning (equal-width bins in this example)
    X_train_transformed[col], bins = pd.cut(
        X_train[col], 
        bins=n_bins[col], 
        labels=False, 
        retbins=True
    )

    X_test_transformed[col] = pd.cut(
        X_test[col],
        bins=bins,      # Use the same bins from X_train
        labels=False,   # Keep consistent labels
        include_lowest=True  # Ensure the lowest bin includes its boundary
    )

    X_train_transformed[col] = X_train_transformed[col].fillna(-1).astype(int)
    X_test_transformed[col] = X_test_transformed[col].fillna(-1).astype(int)

    column_bins[col] = bins

  def get_complex_candidate_target_indices(df, candidate):
    columns, values, conditions = candidate
    mask = pd.Series(True, index=df.index)  # Start with a mask that selects all rows

    for col, val, cond in zip(columns, values, conditions):
      if cond == "<":
        mask &= df.iloc[:, col] < val
      elif cond == "=":
        mask &= df.iloc[:, col] == val
      elif cond == ">":
        mask &= df.iloc[:, col] > val
      else:
        raise ValueError(f"Unsupported condition: {cond}")

    return df[mask].index.tolist()

  pattern1 = p1
  pattern2 = p2
  pattern3 = p3

  target_indices_of_pattern1 = get_complex_candidate_target_indices(X_train_transformed, pattern1)
  target_indices_of_pattern2 = get_complex_candidate_target_indices(X_train_transformed, pattern2)
  target_indices_of_pattern3 = get_complex_candidate_target_indices(X_train_transformed, pattern3)

  uncertain_radius = 0.25*(y_train.max() - y_train.min())
  uncertain_radii = [uncertain_radius, uncertain_radius, uncertain_radius, uncertain_radius, uncertain_radius, uncertain_radius, uncertain_radius]

  uncertain_percentage = 0.1
  uncertain_num = int(uncertain_percentage*len(y_train))
  uncertain_numbers = [uncertain_num, uncertain_num, uncertain_num, uncertain_num, uncertain_num, uncertain_num, uncertain_num]

  dataset_sizes = [len(y_train), len(y_train), len(y_train), len(y_train), len(y_train), len(y_train), len(y_train)]
  dataset_names = [f"{name}_histo(LinReg, MAE)", f"{name}_histo(LinReg, MSE)", f"{name}_histo(Rndfrst, MAE)", f"{name}_histo(Rndfrst, MSE)", f"{name}_Pattern_Mining(Pattern 1)", f"{name}_Pattern_Mining(Pattern 2)", f"{name}_Pattern_Mining(Pattern 3)"]

  dataset_dct = {}
  for i in dataset_names:
    dataset_dct[i] = [X_train, X_test, y_train, y_test]

  boundary_indices_lst = [boundary_indices_lr_mae, boundary_indices_lr_mse, boundary_indices_rf_mae, boundary_indices_rf_mse, target_indices_of_pattern1, target_indices_of_pattern2, target_indices_of_pattern3]

  
  def robustness_score_normalization(uncertain_numbers, uncertain_radii, dataset_sizes, boundary_indices_lst, dataset_names, dataset_dct):
    robustness_radii_10 = [] #find robustness radius that grants radii robustness ratio of 0.10 or more 
                             #(alt. use 0.5 instead {depending on closeness, this ratio may need to be larger})

    for i in range(0, len(dataset_names)):
      uncertain_number = uncertain_numbers[i]
      uncertain_radius = uncertain_radii[i]
      boundary_indices = boundary_indices_lst[i]
      X_train, X_test, y_train, y_test = dataset_dct[dataset_names[i]]
        
      robustness_radius= r_radius
      radius_increment = r_radius_increment

      robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_number,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius,
                                                                    interval=False)
      #print(f"Iteration {i + 1}")  
      #print(robustness_ratio)
      print("Calculating best radius for " + dataset_names[i])
      with tqdm(total=500, desc=f"Finding radius for {dataset_names[i]}", leave=False) as pbar:
          while robustness_ratio < 0.5:
              robustness_radius += radius_increment
              robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_number,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius,
                                                                    interval=False)
              #print(robustness_ratio)
              pbar.update(radius_increment)
        
      robustness_radii_10.append(robustness_radius)

    results = {}
    mean_radius = np.mean(robustness_radii_10)
    std_radius = np.std(robustness_radii_10)
    
    for i, dataset_name in enumerate(dataset_names):
      #normalized_radius = (1 - (robustness_radii_10[i]/max(robustness_radii_10)))
      #normalized_size = (dataset_sizes[i]/max(dataset_sizes))
      robustness_score = 1 - ((robustness_radii_10[i] - mean_radius) / (std_radius + 1e-8))     
      #print(f"Normalized robustness score for {dataset_name} dataset is {robustness_score:.4f}")
      results[dataset_name] = robustness_score

    items = list(sorted(results.items(), key=lambda x: x[1]))
    for item in items:
      print(f"Normalized robustness score for {item[0]} dataset is {item[1]:.4f}")

    worst = items[0]

    print("")
    print(f"So the best targeting approach or the worst case scenario in terms of error injection on the {name} dataset is {worst[0]} with a normalized robustness score of {worst[1]:.4f}")
    print("")
    
    return results

  result_dict = robustness_score_normalization(uncertain_numbers, uncertain_radii, dataset_sizes, boundary_indices_lst, dataset_names, dataset_dct)



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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    for seed in range(1):
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
    df2 = sum([pd.DataFrame(robustness_dicts_interval_lr_mse[i]).iloc[:, 2:] for i in range(1)])/1
    df3 = sum([pd.DataFrame(robustness_dicts_naive[i]).iloc[:, 2:] for i in range(5)])/5  
    df4 = sum([pd.DataFrame(robustness_dicts_lr_mse[i]).iloc[:, 2:] for i in range(1)])/1
    df5 = sum([pd.DataFrame(robustness_dicts_interval_lr_mae[i]).iloc[:, 2:] for i in range(1)])/1
    df6 = sum([pd.DataFrame(robustness_dicts_lr_mae[i]).iloc[:, 2:] for i in range(1)])/1
    df7 = sum([pd.DataFrame(robustness_dicts_interval_rf_mse[i]).iloc[:, 2:] for i in range(1)])/1  
    df8 = sum([pd.DataFrame(robustness_dicts_rf_mse[i]).iloc[:, 2:] for i in range(1)])/1
    df9 = sum([pd.DataFrame(robustness_dicts_interval_rf_mae[i]).iloc[:, 2:] for i in range(1)])/1  
    df10 = sum([pd.DataFrame(robustness_dicts_rf_mae[i]).iloc[:, 2:] for i in range(1)])/1

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
    parser.add_argument('--task', choices=['Pattern_Mining', 'Pattern_Testing_Flawed', 'Pattern_Testing_Set_Percent', 'Normalization', 'leave_one_out'], help="Task to run on terminal: 'Pattern_Mining', 'Pattern_Testing_Flawed', 'Pattern_Testing_Set_Percent', 'Normalization', 'leave_one_out'")
    parser.add_argument("--dataset", choices=["mpg", "ins", "bos", "fire", "all"], help="Name of dataset utilized: 'mpg', 'ins', 'bos', 'fire', 'all'")
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
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 2, maximize=False)
        elif args.dataset == "fire":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, args, ratios, 50, maximize=False)
        else:
            print("")
            print("Dataset is not provided, please provide dataset.")
    elif args.task == "Normalization":
        if args.dataset == "mpg":
            #rndFrst_mae, rndFrst_mse, linReg_mae, linReg_mse
            #lr Mae, lr Mse, rf Mae, rf Mse
            bin_numbers = [44, 23, 44, 44] #fixed, check: 0.25, 0.1, 0.5
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['displacement', 'horsepower', 'weight', 'acceleration']
            num_bins = {'displacement': int(np.sqrt(72)), 'horsepower': int(np.sqrt(87)), 'weight': int(np.sqrt(288)), 'acceleration': int(np.sqrt(86))}
            pattern1 =  ((3, 4, 5), (9, 3, 82.0), ('<', '<', '<')) #0.1
            pattern2 = ((1, 3, 6), (0, 3, 1.0), ('>', '>', '>'))
            pattern3 = ((3, 4, 5), (4, 1, 78.0), ('>', '>', '>'))
            normalization(X_train, y_train, X_test, y_test, bin_numbers, "MPG", 1, 0.01, columns_to_bin, num_bins, pattern1, pattern2, pattern3)
        elif args.dataset == "ins":
            bin_numbers = [15, 36, 41, 38]
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['bmi']
            num_bins = {'bmi': int(np.sqrt(496))}
            pattern1 =  ((1,), (9,), ('=',))
            pattern2 = ((1,), (4,), ('<',))
            pattern3 = ((1,), (10,), ('=',))
            normalization(X_train, y_train, X_test, y_test, bin_numbers, "INS", 1, 100, columns_to_bin, num_bins, pattern1, pattern2, pattern3)
        elif args.dataset == "bos":
            bin_numbers = [7, 7, 22, 7] #fix
            X_train, X_test, y_train, y_test = load_Boston_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']
            num_bins = {'CRIM': int(np.sqrt(402)), 'INDUS': int(np.sqrt(72)), 'NOX': int(np.sqrt(76)), 'RM': int(np.sqrt(366)), 'AGE': int(np.sqrt(302)), 'DIS': int(np.sqrt(339)), 'B': int(np.sqrt(287)), 'LSTAT': int(np.sqrt(366))}
            pattern1 =  ((5, 10), (10, 17.9), ('>', '>'))
            pattern2 = ((7, 12), (3, 5), ('<', '<'))
            pattern3 = ((9, 12), (398.0, 6), ('>', '<'))
            normalization(X_train, y_train, X_test, y_test, bin_numbers, "Boston", 2, 0.01, columns_to_bin, num_bins, pattern1, pattern2, pattern3)
        elif args.dataset == "fire":
            bin_numbers = [9, 9, 5, 9] #fix
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
            num_bins = {'FFMC': int(np.sqrt(103)), 'DMC': int(np.sqrt(199)), 'DC': int(np.sqrt(199)), 'ISI': int(np.sqrt(112)), 'temp': int(np.sqrt(183)), 'RH': int(np.sqrt(73))}
            pattern1 =  ((6, 7), (5, 6), ('=', '<')) #17.3, 0.1 percentage
            pattern2 = ((7, 9), (4, 1.4), ('>', '<'))
            pattern3 = ((5, 8), (0, 4.5), ('=', '<'))
            normalization(X_train, y_train, X_test, y_test, bin_numbers, "Fire", 50, 0.5, columns_to_bin, num_bins, pattern1, pattern2, pattern3)
        elif args.dataset == "all":
            X_train_ins, X_test_ins, y_train_ins, y_test_ins = load_ins_cleaned(random_seed=params["random_seed"])
            X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = load_mpg_cleaned(random_seed=params["random_seed"])
            X_train_bos, X_test_bos, y_train_bos, y_test_bos = load_Boston_cleaned(random_seed=params["random_seed"])
            X_train_fire, X_test_fire, y_train_fire, y_test_fire = load_Fire_cleaned(random_seed=params["random_seed"])
            normalization_all(X_train_ins, X_test_ins, y_train_ins, y_test_ins, X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg, X_train_bos, X_test_bos, y_train_bos, y_test_bos, X_train_fire, X_test_fire, y_train_fire, y_test_fire)
        else:
            print("")
            print("Dataset is not provided, please provided dataset.")
    elif args.task == "Pattern_Testing_Flawed":
        if args.dataset == "mpg":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['displacement', 'horsepower', 'weight', 'acceleration']
            num_bins = {'displacement': int(np.sqrt(72)), 'horsepower': int(np.sqrt(87)), 'weight': int(np.sqrt(288)), 'acceleration': int(np.sqrt(86))}
            pattern1 =  ((3, 4, 5), (9, 3, 82.0), ('<', '<', '<')) #0.1
            pattern2 = ((1, 3, 6), (0, 3, 1.0), ('>', '>', '>'))
            pattern3 = ((3, 4, 5), (4, 1, 78.0), ('>', '>', '>'))
            pattern_testing_heat(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 2, output_dir, args)
        elif args.dataset == "ins":
            ratios = [0.02, 0.04, 0.06, 0.08]
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['bmi']
            num_bins = {'bmi': int(np.sqrt(496))}
            pattern1 =  ((1,), (9,), ('=',))
            pattern2 = ((1,), (4,), ('<',))
            pattern3 = ((1,), (10,), ('=',))
            pattern_testing_heat(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 500, output_dir, args)
        elif args.dataset == "bos":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_Boston_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']
            num_bins = {'CRIM': int(np.sqrt(402)), 'INDUS': int(np.sqrt(72)), 'NOX': int(np.sqrt(76)), 'RM': int(np.sqrt(366)), 'AGE': int(np.sqrt(302)), 'DIS': int(np.sqrt(339)), 'B': int(np.sqrt(287)), 'LSTAT': int(np.sqrt(366))}
            pattern1 =  ((5, 10), (10, 17.9), ('>', '>'))
            pattern2 = ((7, 12), (3, 5), ('<', '<'))
            pattern3 = ((9, 12), (398.0, 6), ('>', '<'))
            pattern_testing_heat(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 2, output_dir, args)
        elif args.dataset == "fire":
            ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
            num_bins = {'FFMC': int(np.sqrt(103)), 'DMC': int(np.sqrt(199)), 'DC': int(np.sqrt(199)), 'ISI': int(np.sqrt(112)), 'temp': int(np.sqrt(183)), 'RH': int(np.sqrt(73))}
            pattern1 =  ((6, 7), (5, 6), ('=', '<')) #17.3, 0.1 percentage
            pattern2 = ((7, 9), (4, 1.4), ('>', '<'))
            pattern3 = ((5, 8), (0, 4.5), ('=', '<'))
            pattern_testing_heat(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 50, output_dir, args)
        else:
            print("")
            print("Dataset is not provided, please provided dataset.")
    elif args.task == "Pattern_Testing_Set_Percent":
        if args.dataset == "mpg":
            ratios = [0.00001, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65] # Trying different range of ratios
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['displacement', 'horsepower', 'weight', 'acceleration']
            num_bins = {'displacement': int(np.sqrt(72)), 'horsepower': int(np.sqrt(87)), 'weight': int(np.sqrt(288)), 'acceleration': int(np.sqrt(86))}
            pattern1 =  ((3, 4, 5), (9, 3, 82.0), ('<', '<', '<')) #0.1
            pattern2 = ((1, 3, 6), (0, 3, 1.0), ('>', '>', '>'))
            pattern3 = ((3, 4, 5), (4, 1, 78.0), ('>', '>', '>'))
            thresholds = [0.275, 0.55, 0.825, 1.1]
            pattern_testing_line(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 2, output_dir, args, thresholds)
        elif args.dataset == "ins":
            ratios = [0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['bmi']
            num_bins = {'bmi': int(np.sqrt(496))}
            pattern1 =  ((1,), (9,), ('=',))
            pattern2 = ((1,), (4,), ('<',))
            pattern3 = ((1,), (10,), ('=',))
            thresholds = [0.04, 0.08, 0.12, 0.16]
            pattern_testing_line(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 500, output_dir, args, thresholds)
        elif args.dataset == "bos":
            ratios = [0.0001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            X_train, X_test, y_train, y_test = load_Boston_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']
            num_bins = {'CRIM': int(np.sqrt(402)), 'INDUS': int(np.sqrt(72)), 'NOX': int(np.sqrt(76)), 'RM': int(np.sqrt(366)), 'AGE': int(np.sqrt(302)), 'DIS': int(np.sqrt(339)), 'B': int(np.sqrt(287)), 'LSTAT': int(np.sqrt(366))}
            pattern1 =  ((5, 10), (10, 17.9), ('>', '>'))
            pattern2 = ((7, 12), (3, 5), ('<', '<'))
            pattern3 = ((9, 12), (398.0, 6), ('>', '<'))
            thresholds = [0.15, 0.3, 0.45, 0.6]
            pattern_testing_line(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 2, output_dir, args, thresholds)
        elif args.dataset == "fire":
            ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
            num_bins = {'FFMC': int(np.sqrt(103)), 'DMC': int(np.sqrt(199)), 'DC': int(np.sqrt(199)), 'ISI': int(np.sqrt(112)), 'temp': int(np.sqrt(183)), 'RH': int(np.sqrt(73))}
            pattern1 =  ((6, 7), (5, 6), ('=', '<')) #17.3, 0.1 percentage
            pattern2 = ((7, 9), (4, 1.4), ('>', '<'))
            pattern3 = ((5, 8), (0, 4.5), ('=', '<'))
            thresholds = [0.125, 0.25, 0.375, 0.5]
            pattern_testing_line(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, ratios, pattern1, pattern2, pattern3, 50, output_dir, args, thresholds)
        else:
            print("")
            print("Dataset is not provided, please provided dataset.")
    elif args.task == "Pattern_Mining":
        if args.dataset == "mpg":
            X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['displacement', 'horsepower', 'weight', 'acceleration']
            num_bins = {'displacement': int(np.sqrt(72)), 'horsepower': int(np.sqrt(87)), 'weight': int(np.sqrt(288)), 'acceleration': int(np.sqrt(86))}
            pattern_mining(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, 3, 2, 0.25, thresholds=[0, 3000], args=None)
        elif args.dataset == "ins":
            X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['bmi']
            num_bins = {'bmi': int(np.sqrt(496))}
            pattern_mining(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, 1, 500, 0.08, thresholds=None, args=None)
        elif args.dataset == "bos":
            X_train, X_test, y_train, y_test = load_Boston_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'B', 'LSTAT']
            num_bins = {'CRIM': int(np.sqrt(402)), 'INDUS': int(np.sqrt(72)), 'NOX': int(np.sqrt(76)), 'RM': int(np.sqrt(366)), 'AGE': int(np.sqrt(302)), 'DIS': int(np.sqrt(339)), 'B': int(np.sqrt(287)), 'LSTAT': int(np.sqrt(366))}
            pattern_mining(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, 2, 2, 0.25, thresholds=[0, 1000], args=True)
        elif args.dataset == "fire":
            X_train, X_test, y_train, y_test = load_Fire_cleaned(random_seed=params["random_seed"])
            columns_to_bin = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
            num_bins = {'FFMC': int(np.sqrt(103)), 'DMC': int(np.sqrt(199)), 'DC': int(np.sqrt(199)), 'ISI': int(np.sqrt(112)), 'temp': int(np.sqrt(183)), 'RH': int(np.sqrt(73))}
            pattern_mining(X_train, X_test, y_train, y_test, columns_to_bin, num_bins, 2, 50, 0.25, thresholds=[3000, 5050], args=None) #Run this to test when terminal is completely fresh deadline wise
    else:
            print("")
            print("No task method given. Please declare type of task to run")


if __name__ == "__main__":
  main()


