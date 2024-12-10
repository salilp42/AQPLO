#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Validation Method Comparison Experiment

This script performs a comprehensive comparison of various cross-validation methods
across multiple datasets and classifiers. It includes data loading, preprocessing,
model training, evaluation, statistical analysis, and the generation of publication-quality figures.

Dependencies:
-------------
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- tqdm
- psutil

Ensure that the 'quicksort_cvs' module is available in the project directory.

Author:
-------
Your Name
Date:
-----
2024-12-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_openml
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
import time
from datetime import datetime
import itertools
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict
import psutil
from typing import Dict, List, Tuple, Any
import os
import traceback
from statsmodels.stats.multitest import multipletests

# Import custom cross-validation methods
from quicksort_cvs import (
    quicksort_leave_pair_out,
    adaptive_quicksort_leave_pair_out
)

# ================================
# Configuration and Setup
# ================================

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.dpi': 800,
    'savefig.dpi': 800,
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'legend.fontsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'axes.axisbelow': True
})

# Create necessary directories to store results
for dir_name in ['results', 'results/figures', 'results/tables', 
                 'results/statistics', 'results/raw_data']:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# ================================
# Data Generation and Loading
# ================================

def generate_synthetic_data(n_samples: int = 1000,
                           n_features: int = 20,
                           noise_level: float = 0.1,
                           group_effect_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with controlled properties for benchmarking.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features per sample
    noise_level : float
        Amount of random noise to add
    group_effect_strength : float
        Strength of group-specific effects

    Returns:
    --------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Binary target values
    groups : np.ndarray
        Group assignments
    """
    # Generate base features from a standard normal distribution
    X = np.random.randn(n_samples, n_features)
    
    # Determine the number of groups based on the number of samples
    n_groups = min(10, n_samples // 50)
    groups = np.zeros(n_samples)
    samples_per_group = n_samples // n_groups
    
    # Generate group-specific effects
    group_effects = np.random.randn(n_groups, n_features) * group_effect_strength
    
    y = np.zeros(n_samples)
    for i in range(n_groups):
        start_idx = i * samples_per_group
        end_idx = start_idx + samples_per_group if i < n_groups - 1 else n_samples
        
        # Assign group labels
        groups[start_idx:end_idx] = i
        
        # Add group-specific effects to features
        X[start_idx:end_idx] += group_effects[i]
        
        # Generate target values based on a logistic function
        logits = np.dot(X[start_idx:end_idx], np.random.randn(n_features))
        logits += group_effect_strength * np.random.randn(end_idx - start_idx)
        probs = 1 / (1 + np.exp(-logits))
        y[start_idx:end_idx] = (probs > 0.5).astype(int)
    
    # Add random noise to the features
    X += np.random.randn(*X.shape) * noise_level
    
    return X, y, groups.astype(int)

def load_datasets() -> Dict[str, Dict[str, Any]]:
    """
    Load and prepare all datasets for the experiment.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary containing all prepared datasets
    """
    print("Loading datasets...")
    datasets = {}
    
    try:
        # Load Breast Cancer dataset
        X_bc, y_bc = load_breast_cancer(return_X_y=True)
        datasets['Breast Cancer'] = {
            'X': X_bc,
            'y': y_bc,
            'groups': None
        }
        print("Loaded Breast Cancer dataset")
        
        # Create a high-dimensional version using polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_bc_high = poly.fit_transform(X_bc)
        datasets['Breast Cancer (High Dim)'] = {
            'X': X_bc_high,
            'y': y_bc,
            'groups': None
        }
        print("Created high-dimensional version of Breast Cancer dataset")
        
        # Load Diabetes dataset
        X_db, y_db = load_diabetes(return_X_y=True)
        datasets['Diabetes'] = {
            'X': X_db,
            'y': y_db,
            'groups': None
        }
        print("Loaded Diabetes dataset")
        
        # Load Ionosphere dataset from OpenML
        X_ion, y_ion = fetch_openml(name='ionosphere', version=1, return_X_y=True, as_frame=False)
        datasets['Ionosphere'] = {
            'X': X_ion,
            'y': y_ion,
            'groups': None
        }
        print("Loaded Ionosphere dataset")
        
        # Generate synthetic dataset
        print("Generating synthetic dataset...")
        X_syn, y_syn, groups_syn = generate_synthetic_data()
        datasets['Synthetic'] = {
            'X': X_syn,
            'y': y_syn,
            'groups': groups_syn
        }
        print("Generated Synthetic dataset")
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        print(traceback.format_exc())
        raise
    
    return datasets

# ================================
# Data Preprocessing
# ================================

def preprocess_dataset(X: np.ndarray,
                      y: np.ndarray,
                      dataset_name: str,
                      n_groups: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess dataset with enhanced group creation and proper synthetic data handling.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    dataset_name : str
        Name of the dataset
    n_groups : int
        Target number of groups (may be adjusted based on dataset size)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Preprocessed X, y, and group assignments
    """
    # Convert target to binary for non-binary datasets
    if dataset_name == 'Diabetes':
        y = (y > np.median(y)).astype(int)
    elif dataset_name == 'Ionosphere':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create balanced groups based on the dataset
    if dataset_name == 'Synthetic':
        # For synthetic data, ensure balanced groups
        n_groups_actual = min(n_groups, len(X) // 50)  # Ensure enough samples per group
        samples_per_group = len(X) // n_groups_actual
        groups = np.array([i // samples_per_group for i in range(len(X))])
        groups[groups >= n_groups_actual] = n_groups_actual - 1  # Handle any remainder
    else:
        # For real datasets, create stratified groups
        groups = np.zeros(len(y))
        n_groups_actual = min(n_groups, len(X) // 20)  # Ensure enough samples per group
        
        # Stratify by class to maintain class balance within groups
        for label in np.unique(y):
            mask = (y == label)
            n_samples = np.sum(mask)
            
            # Calculate samples per group
            base_samples = n_samples // n_groups_actual
            remainder = n_samples % n_groups_actual
            
            # Distribute remainder samples
            group_sizes = np.repeat(base_samples, n_groups_actual)
            group_sizes[:remainder] += 1
            
            # Assign groups and shuffle
            group_assignments = np.repeat(np.arange(n_groups_actual), group_sizes)
            np.random.shuffle(group_assignments)
            groups[mask] = group_assignments
    
    print(f"Created {len(np.unique(groups))} groups with sizes: "
          f"{np.bincount(groups.astype(int))}")
    
    return X_scaled, y, groups.astype(int)

# ================================
# Cross-Validation Methods
# ================================

def tournament_leave_pair_out(X: np.ndarray,
                               y: np.ndarray,
                               groups: np.ndarray) -> tuple:
    """
    Generate training/testing splits using tournament-style leave-pair-out cross-validation.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    groups : np.ndarray
        Group labels for each sample

    Yields:
    -------
    tuple
        (train_indices, test_indices) for each split
    """
    unique_groups = np.unique(groups)
    
    # Generate all possible pairs of groups
    for i, j in itertools.combinations(range(len(unique_groups)), 2):
        group_i, group_j = unique_groups[i], unique_groups[j]
        
        # Create masks for training and testing
        train_mask = ~np.isin(groups, [group_i, group_j])
        test_mask = ~train_mask
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Ensure both splits have sufficient class diversity
        if (len(np.unique(y[train_indices])) < 2 or 
            len(np.unique(y[test_indices])) < 2):
            continue
            
        yield train_indices, test_indices

# ================================
# Evaluation Metrics
# ================================

def calculate_confidence_intervals(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   n_bootstraps: int = 1000,
                                   ci_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate both bootstrap and DeLong confidence intervals for AUC.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted probabilities
    n_bootstraps : int
        Number of bootstrap samples
    ci_level : float
        Confidence level (0-1)

    Returns:
    --------
    Dict[str, Tuple[float, float]]
        Dictionary containing both types of confidence intervals
    """
    results = {}
    
    try:
        # Bootstrap Confidence Interval
        bootstrap_aucs = []
        rng = np.random.RandomState(42)
        
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                continue
            bootstrap_aucs.append(roc_auc_score(y_true[indices], y_pred[indices]))
        
        bootstrap_aucs = np.array(bootstrap_aucs)
        alpha = (1 - ci_level) / 2
        bootstrap_ci = (np.percentile(bootstrap_aucs, 100 * alpha),
                       np.percentile(bootstrap_aucs, 100 * (1 - alpha)))
        
        results['bootstrap'] = bootstrap_ci
        
        # DeLong Confidence Interval
        auc = roc_auc_score(y_true, y_pred)
        n_pos = np.sum(y_true == 1)
        n_neg = len(y_true) - n_pos
        
        # Calculate standard error using DeLong's method approximation
        q1 = np.zeros(n_pos)
        q2 = np.zeros(n_neg)
        
        for i in range(n_pos):
            q1[i] = np.mean(y_pred[y_true == 0] < y_pred[y_true == 1][i])
        
        for i in range(n_neg):
            q2[i] = np.mean(y_pred[y_true == 1] > y_pred[y_true == 0][i])
        
        se = np.sqrt((np.var(q1) / n_pos + np.var(q2) / n_neg))
        
        # Calculate Confidence Interval
        z_value = stats.norm.ppf(1 - (1 - ci_level) / 2)
        delong_ci = (auc - z_value * se, auc + z_value * se)
        
        results['delong'] = delong_ci
        
    except Exception as e:
        print(f"Error calculating confidence intervals: {str(e)}")
        results['bootstrap'] = (np.nan, np.nan)
        results['delong'] = (np.nan, np.nan)
    
    return results

# ================================
# Cross-Validation Execution
# ================================

def run_cv_method(X: np.ndarray,
                 y: np.ndarray,
                 groups: np.ndarray,
                 clf: Any,
                 cv_method: Any,
                 method_name: str,
                 pbar: tqdm) -> Dict[str, Any]:
    """
    Run cross-validation method with comprehensive metrics collection.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    groups : np.ndarray
        Group assignments
    clf : Any
        Classifier object
    cv_method : Any
        Cross-validation method
    method_name : str
        Name of the method
    pbar : tqdm
        Progress bar object

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all results and metrics
    """
    results = {
        'method': method_name,
        'scores': [],
        'predictions': [],
        'true_values': [],
        'timing': [],
        'memory': [],
        'n_samples': X.shape[0],  # Store dataset dimensions
        'n_features': X.shape[1],
        'group_sizes': np.bincount(groups)
    }
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
    
    try:
        # Generate splits based on the cross-validation method
        if method_name in ['QPLO', 'AQPLO']:
            splits = list(cv_method(X, y, groups, clf))
        else:
            splits = list(cv_method(X, y, groups))
        
        # Process each split
        for train_idx, test_idx in splits:
            split_start = time.time()
            
            # Split data into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train the classifier and predict probabilities
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            
            # Calculate AUC score
            score = roc_auc_score(y_test, y_pred)
            
            # Store results
            results['scores'].append(score)
            results['predictions'].extend(y_pred)
            results['true_values'].extend(y_test)
            
            # Track timing and memory
            split_time = time.time() - split_start
            results['timing'].append(split_time)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
            results['memory'].append(current_memory - start_memory)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_description(
                f"{method_name}: AUC={score:.3f} (avg={np.mean(results['scores']):.3f})"
            )
    
    except Exception as e:
        print(f"\nError in {method_name}: {str(e)}")
        print(traceback.format_exc())
        return None
    
    # Calculate final metrics
    results['mean_auc'] = np.mean(results['scores'])
    results['std_auc'] = np.std(results['scores'])
    results['confidence_intervals'] = calculate_confidence_intervals(
        np.array(results['true_values']),
        np.array(results['predictions'])
    )
    
    # Calculate timing statistics
    results['total_time'] = time.time() - start_time
    results['mean_split_time'] = np.mean(results['timing'])
    results['peak_memory'] = max(results['memory']) if results['memory'] else np.nan
    
    return results

# ================================
# Statistical Analysis
# ================================

def calculate_statistical_significance(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis of results with multiple comparisons correction.

    Parameters:
    -----------
    all_results : Dict[str, Dict[str, Dict[str, Any]]]
        Results from all experiments

    Returns:
    --------
    Dict[str, Any]
        Statistical analysis results including corrected p-values
    """
    stats_results = {
        'friedman': {},
        'wilcoxon': defaultdict(dict),
        'effect_sizes': defaultdict(dict),
        'adjusted_p_values': {}
    }
    
    # Define the methods to analyze
    methods = ['GSKF', 'LOGO', 'TLPO', 'QPLO', 'AQPLO']
    performance_data = defaultdict(list)
    
    # Aggregate performance data for each method across datasets and classifiers
    for dataset in all_results:
        for clf in all_results[dataset]:
            for method in methods:
                if method in all_results[dataset][clf] and all_results[dataset][clf][method] is not None:
                    performance_data[method].append(all_results[dataset][clf][method]['mean_auc'])
    
    # Perform Friedman test to check for overall differences
    try:
        performance_arrays = [performance_data[method] for method in methods]
        friedman_stat, friedman_p = friedmanchisquare(*performance_arrays)
        stats_results['friedman'] = {
            'statistic': float(friedman_stat),
            'p_value': float(friedman_p)
        }
    except Exception as e:
        print(f"Error in Friedman test: {str(e)}")
        stats_results['friedman'] = {'error': str(e)}
    
    # Perform pairwise Wilcoxon tests
    wilcoxon_results = []
    comparisons = list(itertools.combinations(methods, 2))
    for m1, m2 in comparisons:
        try:
            stat, p_value = wilcoxon(performance_data[m1], performance_data[m2])
            wilcoxon_results.append((f'{m1}_vs_{m2}', stat, p_value))
            
            # Calculate effect size (Cohen's d)
            d = (np.mean(performance_data[m1]) - np.mean(performance_data[m2])) / \
                np.sqrt((np.var(performance_data[m1], ddof=1) + np.var(performance_data[m2], ddof=1)) / 2)
            
            stats_results['wilcoxon'][f'{m1}_vs_{m2}'] = {
                'statistic': float(stat),
                'p_value': float(p_value)
            }
            stats_results['effect_sizes'][f'{m1}_vs_{m2}'] = float(d)
        except Exception as e:
            print(f"Error in comparison {m1} vs {m2}: {str(e)}")
            stats_results['wilcoxon'][f'{m1}_vs_{m2}'] = {'error': str(e)}
            stats_results['effect_sizes'][f'{m1}_vs_{m2}'] = np.nan
    
    # Extract p-values for multiple testing correction
    p_values = [result[2] for result in wilcoxon_results]
    comparison_names = [result[0] for result in wilcoxon_results]
    
    # Apply Benjamini-Hochberg FDR correction
    try:
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        stats_results['adjusted_p_values'] = dict(zip(comparison_names, corrected_p))
        
        # Update Wilcoxon results with adjusted p-values and significance flags
        for i, comp in enumerate(comparison_names):
            if 'wilcoxon' in stats_results and comp in stats_results['wilcoxon']:
                stats_results['wilcoxon'][comp]['adjusted_p_value'] = float(corrected_p[i])
                stats_results['wilcoxon'][comp]['significant'] = bool(rejected[i])
    except Exception as e:
        print(f"Error in p-value correction: {str(e)}")
        stats_results['adjusted_p_values'] = {'error': str(e)}
    
    return stats_results

# ================================
# Result Saving and Encoding
# ================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

def save_results(all_results: Dict[str, Dict[str, Dict[str, Any]]], 
                stats_results: Dict[str, Any]):
    """
    Save all results to files.

    Parameters:
    -----------
    all_results : Dict[str, Dict[str, Dict[str, Any]]]
        Results from all experiments
    stats_results : Dict[str, Any]
        Statistical analysis results
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results as JSON
    with open(f'results/raw_data/all_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, cls=NumpyEncoder)
    
    # Save statistical analysis results as JSON
    with open(f'results/statistics/stats_results_{timestamp}.json', 'w') as f:
        json.dump(stats_results, f, cls=NumpyEncoder)
    
    # Create a summary CSV file
    summary_data = []
    for dataset in all_results:
        for clf in all_results[dataset]:
            for method in all_results[dataset][clf]:
                if all_results[dataset][clf][method] is not None:
                    result = all_results[dataset][clf][method]
                    summary_data.append({
                        'Dataset': dataset,
                        'Classifier': clf,
                        'Method': method,
                        'AUC': result['mean_auc'],
                        'Std': result['std_auc'],
                        'Time': result['total_time'],
                        'Memory': result['peak_memory'],
                        'CI_Lower': result['confidence_intervals']['delong'][0],
                        'CI_Upper': result['confidence_intervals']['delong'][1]
                    })
    
    pd.DataFrame(summary_data).to_csv(f'results/tables/summary_{timestamp}.csv', index=False)

# ================================
# Figure Generation
# ================================

def create_publication_figures(all_results: Dict[str, Dict[str, Dict[str, Any]]],
                              stats_results: Dict[str, Any]):
    """
    Create publication-quality integrated figures for the paper with improved aesthetics and clarity.

    Parameters:
    -----------
    all_results : Dict[str, Dict[str, Dict[str, Any]]]
        All experimental results
    stats_results : Dict[str, Any]
        Statistical analysis results
    """
    try:
        methods = ['GSKF', 'LOGO', 'TLPO', 'QPLO', 'AQPLO']
        pastel_colors = sns.color_palette("pastel", n_colors=len(methods))
        color_dict = dict(zip(methods, pastel_colors))
        
        # Define a pastel sequential colormap for the heatmap
        pastel_sequential_cmap = sns.light_palette("seagreen", as_cmap=True)
        
        # Figure 1: Cross-Validation Method Comparison
        fig1, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig1.suptitle('Cross-Validation Method Comparison', fontsize=20, y=1.02, fontweight='bold')
        
        # Prepare data structures for performance metrics
        perf_data = {method: {'auc': [], 'time': [], 'ci_width': []} for method in methods}
        dataset_sizes = {}
        
        for dataset in all_results:
            for clf in all_results[dataset]:
                for method in methods:
                    if method in all_results[dataset][clf] and all_results[dataset][clf][method]:
                        result = all_results[dataset][clf][method]
                        perf_data[method]['auc'].append(result['mean_auc'])
                        perf_data[method]['time'].append(result['total_time'])
                        ci = result['confidence_intervals']['delong']
                        perf_data[method]['ci_width'].append(ci[1] - ci[0])
                        
                        if dataset not in dataset_sizes:
                            dataset_sizes[dataset] = result['n_samples']
        
        # (a) AUC Performance with Confidence Intervals
        ax = axes[0, 0]
        auc_data = [perf_data[method]['auc'] for method in methods]
        sns.boxplot(data=auc_data, palette=pastel_colors, ax=ax, width=0.6)
        ax.set_xticklabels(methods, rotation=45, fontsize=12)
        ax.set_ylabel('AUC Score', fontsize=14)
        ax.set_title('(a) Classification Performance', fontsize=16, pad=20)
        
        # Add significance indicators if available
        if 'wilcoxon' in stats_results:
            max_auc = max([max(perf_data[m]['auc']) for m in methods if perf_data[m]['auc']])
            y_pos = max_auc + 0.02
            for comp, result in stats_results['wilcoxon'].items():
                if ('p_value' in result and result['adjusted_p_value'] < 0.05):
                    m1, m2 = comp.split('_vs_')
                    idx1, idx2 = methods.index(m1), methods.index(m2)
                    ax.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1)
                    ax.text((idx1 + idx2)/2, y_pos + 0.01, '*', ha='center', va='bottom', fontsize=20)
        
        # (b) Computational Efficiency
        ax = axes[0, 1]
        time_data = [perf_data[method]['time'] for method in methods]
        sns.violinplot(data=time_data, palette=pastel_colors, ax=ax, inner='quartile', cut=0)
        ax.set_xticklabels(methods, rotation=45, fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=14)
        ax.set_title('(b) Computational Efficiency', fontsize=16, pad=20)
        
        # Add median markers
        medians = [np.median(perf_data[method]['time']) for method in methods]
        ax.scatter(range(len(methods)), medians, color='white', zorder=10, label='Median', marker='D')
        ax.legend()
        
        # (c) Confidence Interval Width Analysis
        ax = axes[1, 0]
        ci_width_data = [perf_data[method]['ci_width'] for method in methods]
        sns.boxplot(data=ci_width_data, palette=pastel_colors, ax=ax, width=0.6)
        ax.set_xticklabels(methods, rotation=45, fontsize=12)
        ax.set_ylabel('CI Width (DeLong)', fontsize=14)
        ax.set_title('(c) Uncertainty Quantification', fontsize=16, pad=20)
        
        # (d) CI Coverage Analysis Using Violin Plot
        ax = axes[1, 1]
        ci_coverage = defaultdict(list)
        for dataset in all_results:
            for clf in all_results[dataset]:
                for method in methods:
                    if method in all_results[dataset][clf] and all_results[dataset][clf][method]:
                        result = all_results[dataset][clf][method]
                        true_auc = result['mean_auc']
                        ci = result['confidence_intervals']['delong']
                        ci_coverage[method].append(
                            1 if ci[0] <= true_auc <= ci[1] else 0
                        )
        
        coverage_data = [ci_coverage[method] for method in methods]
        sns.violinplot(data=coverage_data, palette=pastel_colors, ax=ax, inner='quartile', cut=0)
        ax.set_xticklabels(methods, rotation=45, fontsize=12)
        ax.set_ylabel('CI Coverage Rate', fontsize=14)
        ax.set_title('(d) CI Coverage Analysis', fontsize=16, pad=20)
        
        # Overlay a swarm plot for individual coverage rates
        sns.swarmplot(data=coverage_data, color='0.2', ax=ax, size=3)
        ax.axhline(y=0.95, color='red', linestyle='--', label='95% Target Coverage')
        ax.legend()
        
        # Adjust layout and save Figure 1
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('results/figures/integrated_performance_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Detailed Method Analysis
        fig2, axes = plt.subplots(2, 1, figsize=(18, 14))
        fig2.suptitle('Detailed Method Analysis', fontsize=20, y=1.02, fontweight='bold')
        
        # (a) Per-Dataset Performance Heatmap
        ax = axes[0]
        dataset_perf = defaultdict(lambda: defaultdict(list))
        for dataset in all_results:
            for clf in all_results[dataset]:
                for method in methods:
                    if method in all_results[dataset][clf] and all_results[dataset][clf][method]:
                        dataset_perf[dataset][method].append(
                            all_results[dataset][clf][method]['mean_auc']
                        )
        
        # Create heatmap data
        heatmap_data = pd.DataFrame(
            {method: [np.mean(dataset_perf[dataset][method]) for dataset in dataset_perf]
             for method in methods},
            index=list(dataset_perf.keys())
        )
        
        # Generate a full-spectrum pastel colormap
        sns.heatmap(heatmap_data, ax=ax, cmap=pastel_sequential_cmap, annot=True, fmt='.3f',
                    cbar_kws={'label': 'Mean AUC Score'}, linewidths=0.5, linecolor='gray')
        ax.set_xlabel('Methods', fontsize=14)
        ax.set_ylabel('Datasets', fontsize=14)
        ax.set_title('(a) Per-Dataset Performance', fontsize=16, pad=20)
        
        # (b) Dimensionality Impact Analysis
        ax = axes[1]
        dim_effects = defaultdict(lambda: {'dims': [], 'auc': []})
        
        for dataset in all_results:
            for clf in all_results[dataset]:
                for method in methods:
                    if method in all_results[dataset][clf] and all_results[dataset][clf][method]:
                        result = all_results[dataset][clf][method]
                        dims = result['n_features']
                        auc = result['mean_auc']
                        dim_effects[method]['dims'].append(dims)
                        dim_effects[method]['auc'].append(auc)
        
        for method in methods:
            if dim_effects[method]['dims']:  # Check if data is available
                sns.scatterplot(x=dim_effects[method]['dims'], 
                                y=dim_effects[method]['auc'],
                                color=color_dict[method], label=method, alpha=0.7, ax=ax)
                
                # Add trend line if enough points are available
                if len(dim_effects[method]['dims']) > 1:
                    z = np.polyfit(dim_effects[method]['dims'], 
                                  dim_effects[method]['auc'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(dim_effects[method]['dims']), 
                                          max(dim_effects[method]['dims']), 100)
                    ax.plot(x_trend, p(x_trend), '--', color=color_dict[method], alpha=0.5)
        
        ax.set_xlabel('Number of Features', fontsize=14)
        ax.set_ylabel('AUC Score', fontsize=14)
        ax.set_title('(b) Dimensionality Impact', fontsize=16, pad=20)
        ax.legend(title='Methods')
        
        # Adjust layout and save Figure 2
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('results/figures/detailed_method_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Method Performance Summary
        fig3, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig3.suptitle('Method Performance Summary', fontsize=20, y=1.02, fontweight='bold')
        
        # (a) Runtime vs AUC Trade-off
        ax = axes[0]
        for method in methods:
            mean_auc = np.mean(perf_data[method]['auc'])
            mean_time = np.mean(perf_data[method]['time'])
            std_auc = np.std(perf_data[method]['auc'])
            std_time = np.std(perf_data[method]['time'])
            
            ax.errorbar(mean_time, mean_auc, 
                        xerr=std_time, yerr=std_auc,
                        fmt='o', color=color_dict[method], label=method,
                        capsize=5, markersize=10, alpha=0.8)
        
        ax.set_xlabel('Average Runtime (seconds)', fontsize=14)
        ax.set_ylabel('Average AUC Score', fontsize=14)
        ax.set_title('(a) Performance-Runtime Trade-off', fontsize=16, pad=20)
        ax.legend(title='Methods')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # (b) Scalability Analysis
        ax = axes[1]
        for method in methods:
            sizes = []
            times = []
            for dataset, size in dataset_sizes.items():
                for clf in all_results[dataset]:
                    if method in all_results[dataset][clf] and all_results[dataset][clf][method]:
                        result = all_results[dataset][clf][method]
                        sizes.append(size)
                        times.append(result['total_time'])
            
            if len(sizes) > 1:
                sns.scatterplot(x=sizes, y=times, color=color_dict[method], label=method, alpha=0.7, ax=ax)
                
                # Fit and plot trend line
                z = np.polyfit(sizes, times, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(sizes), max(sizes), 100)
                ax.plot(x_trend, p(x_trend), '--', color=color_dict[method], alpha=0.5)
        
        ax.set_xlabel('Dataset Size (samples)', fontsize=14)
        ax.set_ylabel('Total Runtime (seconds)', fontsize=14)
        ax.set_title('(b) Scalability Analysis', fontsize=16, pad=20)
        ax.legend(title='Methods')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Adjust layout and save Figure 3
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('results/figures/method_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating figures: {str(e)}")
        print(traceback.format_exc())
        # Create empty figures to prevent complete failure
        for name in ['integrated_performance_analysis.png', 
                    'detailed_method_analysis.png',
                    'method_summary.png']:
            plt.figure()
            plt.savefig(f'results/figures/{name}')
            plt.close()
    finally:
        plt.close('all')

# ================================
# Main Execution Function
# ================================

def main():
    """Main execution function for the experiment."""
    print("\n=== Starting Cross-validation Method Comparison Experiment ===\n")
    start_time = time.time()
    
    # Suppress warnings but keep important ones
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('always', category=RuntimeWarning)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Load datasets
        datasets = load_datasets()
        print(f"\nLoaded {len(datasets)} datasets successfully")
        
        # Initialize classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        }
        
        # Initialize cross-validation methods
        cv_methods = {
            'GSKF': lambda X, y, groups: GroupKFold(n_splits=5).split(X, y, groups),
            'LOGO': lambda X, y, groups: LeaveOneGroupOut().split(X, y, groups),
            'TLPO': tournament_leave_pair_out,
            'QPLO': quicksort_leave_pair_out,
            'AQPLO': adaptive_quicksort_leave_pair_out
        }
        
        # Initialize storage for all results
        all_results = {}
        execution_times = defaultdict(list)
        memory_usage = defaultdict(list)
        
        # Overall progress bar
        with tqdm(total=len(datasets) * len(classifiers), 
                 desc="Overall Progress", position=0) as pbar_outer:
            
            # Iterate over each dataset
            for dataset_name, dataset in datasets.items():
                print(f"\n{'='*80}")
                print(f"Processing {dataset_name}...")
                print(f"{'='*80}")
                
                all_results[dataset_name] = {}
                
                # Preprocess the dataset
                X, y, groups = preprocess_dataset(dataset['X'], dataset['y'], dataset_name)
                print(f"Dataset shape: {X.shape}, Unique groups: {len(np.unique(groups))}")
                
                # Iterate over each classifier
                for clf_name, clf in classifiers.items():
                    print(f"\nClassifier: {clf_name}")
                    print("-" * 40)
                    
                    all_results[dataset_name][clf_name] = {}
                    
                    # Estimate total splits for progress bar
                    n_splits_estimate = sum(1 for _ in itertools.islice(
                        cv_methods['GSKF'](X, y, groups), 5))
                    total_splits_estimate = n_splits_estimate * len(cv_methods)
                    
                    with tqdm(total=total_splits_estimate, 
                              desc=f"CV Methods ({clf_name})", 
                              position=1, 
                              leave=False) as pbar_inner:
                        
                        # Iterate over each cross-validation method
                        for method_name, cv_method in cv_methods.items():
                            try:
                                method_start = time.time()
                                results = run_cv_method(
                                    X, y, groups, clf, cv_method, 
                                    method_name, pbar_inner
                                )
                                
                                if results is not None:
                                    all_results[dataset_name][clf_name][method_name] = results
                                    execution_times[method_name].append(results['total_time'])
                                    memory_usage[method_name].append(results['peak_memory'])
                                    
                                    # Print method results
                                    print(f"\n{method_name} Results:")
                                    print(f"  AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
                                    print(f"  Time: {results['total_time']:.2f}s")
                                    print(f"  Memory: {results['peak_memory']:.2f}MB")
                                    print(f"  CI (DeLong): [{results['confidence_intervals']['delong'][0]:.4f}, "
                                          f"{results['confidence_intervals']['delong'][1]:.4f}]")
                                
                            except Exception as e:
                                print(f"\nError in {method_name}: {str(e)}")
                                print(traceback.format_exc())
                                continue
                    
                    # Update overall progress bar after processing classifiers
                    pbar_outer.update(1)
        
        # Perform statistical analysis on the collected results
        print("\nPerforming statistical analysis...")
        stats_results = calculate_statistical_significance(all_results)
        
        # Generate publication-quality figures based on the results
        print("\nGenerating publication-quality figures...")
        create_publication_figures(all_results, stats_results)
        
        # Display statistical analysis summary
        print("\nStatistical Analysis Summary:")
        print("-" * 40)
        if 'friedman' in stats_results:
            print(f"Friedman Test:")
            print(f"  Statistic: {stats_results['friedman'].get('statistic', 'N/A'):.4f}")
            print(f"  p-value: {stats_results['friedman'].get('p_value', 'N/A'):.4f}")
        
        print("\nPairwise Comparisons (Wilcoxon):")
        for comparison, result in stats_results['wilcoxon'].items():
            if 'error' not in result:
                print(f"  {comparison}:")
                print(f"    p-value: {result['p_value']:.4f}")
                print(f"    effect size: {stats_results['effect_sizes'][comparison]:.4f}")
        
        # Save all results and statistical analysis
        print("\nSaving results...")
        save_results(all_results, stats_results)
        
        # Create and save a summary table of execution times and memory usage
        summary_data = []
        for method, times in execution_times.items():
            summary_data.append({
                'Method': method,
                'Avg Time (s)': f"{np.mean(times):.2f} ± {np.std(times):.2f}",
                'Avg Memory (MB)': f"{np.mean(memory_usage[method]):.2f} ± {np.std(memory_usage[method]):.2f}",
                'Success Rate': f"{len(times)/float(len(datasets)*len(classifiers)):.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nFinal Performance Summary:")
        print(summary_df.to_string(index=False))
        summary_df.to_csv('results/tables/final_summary.csv', index=False)
        
        # Print total execution time
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        
        print("\n=== Experiment completed successfully ===")
        
    except Exception as e:
        print(f"\nCritical error in experiment execution: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())
        raise
    
    finally:
        # Ensure all figures are closed to free memory
        plt.close('all')

# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    main()
