"""
Capacity planning evaluation for network traffic forecasting.

Compares how different forecasting models affect capacity planning decisions:
- Computes capacities based on predicted 95th percentile loads
- Evaluates utilization and overload metrics against true loads
- Generates comparison plots
"""

import os
import numpy as np

from config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR
from utils import (
    load_json, save_json, print_summary_table,
    plot_rmse_histogram, plot_metric_comparison, plot_capacity_bars,
    plot_timeseries_comparison
)


def load_predictions():
    """
    Load predictions from both models and align them.

    Returns:
        Dictionary with aligned predictions and true values
    """
    # Load SARIMA predictions
    sarima_data = np.load(os.path.join(RESULTS_DIR, 'sarima_predictions.npz'))
    sarima_preds = sarima_data['predictions']
    sarima_L_test = sarima_data['L_test']

    # Load LSTM predictions
    lstm_data = np.load(os.path.join(RESULTS_DIR, 'lstm_predictions.npz'))
    lstm_preds = lstm_data['predictions']
    lstm_L_test = lstm_data['L_test_aligned']

    # SARIMA predicts all T_test steps
    # LSTM predicts T_test - window_size steps (starting at window_size offset)
    window_size = CONFIG['window_size']

    # Align SARIMA predictions to match LSTM
    sarima_preds_aligned = sarima_preds[window_size:]
    sarima_L_test_aligned = sarima_L_test[window_size:]

    # Verify alignment
    assert sarima_preds_aligned.shape == lstm_preds.shape, \
        f"Shape mismatch: SARIMA {sarima_preds_aligned.shape} vs LSTM {lstm_preds.shape}"
    assert np.allclose(sarima_L_test_aligned, lstm_L_test), \
        "True values mismatch between SARIMA and LSTM"

    return {
        'Y_true': lstm_L_test,
        'Y_pred_sarima': sarima_preds_aligned,
        'Y_pred_lstm': lstm_preds,
        'T_eff': lstm_preds.shape[0],
        'num_links': lstm_preds.shape[1]
    }


def compute_capacities(predictions: dict, percentile: float = 95,
                       margin: float = 1.1) -> dict:
    """
    Compute link capacities based on predicted loads.

    For each model, capacity = margin * percentile(predicted_loads)

    Args:
        predictions: Dictionary with aligned predictions
        percentile: Percentile of loads to use (default 95)
        margin: Safety margin multiplier (default 1.1 = 10%)

    Returns:
        Dictionary mapping model name to capacity array
    """
    Y_true = predictions['Y_true']
    Y_pred_sarima = predictions['Y_pred_sarima']
    Y_pred_lstm = predictions['Y_pred_lstm']

    capacities = {}

    # SARIMA capacities
    Q95_sarima = np.percentile(Y_pred_sarima, percentile, axis=0)
    capacities['SARIMA'] = margin * Q95_sarima

    # LSTM capacities
    Q95_lstm = np.percentile(Y_pred_lstm, percentile, axis=0)
    capacities['LSTM'] = margin * Q95_lstm

    # Oracle capacities (using true values)
    Q95_oracle = np.percentile(Y_true, percentile, axis=0)
    capacities['Oracle'] = margin * Q95_oracle

    return capacities


def compute_utilization_metrics(Y_true: np.ndarray, capacities: dict) -> dict:
    """
    Compute utilization and overload metrics for each model.

    Args:
        Y_true: True loads of shape (T, num_links)
        capacities: Dictionary mapping model name to capacity array

    Returns:
        Dictionary mapping model name to metrics dictionary
    """
    metrics = {}

    for model_name, cap in capacities.items():
        # Avoid division by zero
        cap_safe = np.maximum(cap, 1e-6)

        # Per-time-step utilization
        utilization = Y_true / cap_safe  # Shape: (T, num_links)

        # Max utilization per link
        u_max = utilization.max(axis=0)  # Shape: (num_links,)

        # Fraction of time in overload per link
        f_over = (utilization > 1.0).mean(axis=0)  # Shape: (num_links,)

        # Aggregate metrics
        metrics[model_name] = {
            'u_max_per_link': u_max.tolist(),
            'f_over_per_link': f_over.tolist(),
            'u_max_mean': float(u_max.mean()),
            'u_max_max': float(u_max.max()),
            'u_max_median': float(np.median(u_max)),
            'f_over_mean': float(f_over.mean()),
            'f_over_max': float(f_over.max()),
            'links_over_110': int((u_max > 1.1).sum()),  # Badly under-provisioned
            'links_over_100': int((u_max > 1.0).sum()),  # Any overload
        }

    return metrics


def generate_plots(predictions: dict, capacities: dict,
                   forecasting_metrics: dict, capacity_metrics: dict):
    """Generate all comparison plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Y_true = predictions['Y_true']
    Y_pred_sarima = predictions['Y_pred_sarima']
    Y_pred_lstm = predictions['Y_pred_lstm']
    T_eff = predictions['T_eff']

    # 1. RMSE histogram comparison
    print("   - Generating RMSE histogram...")
    sarima_metrics = load_json(os.path.join(RESULTS_DIR, 'sarima_metrics.json'))
    lstm_metrics = load_json(os.path.join(RESULTS_DIR, 'lstm_metrics.json'))

    rmse_dict = {
        'SARIMA': np.array(sarima_metrics['per_link']['rmse']),
        'LSTM': np.array(lstm_metrics['per_link']['rmse'])
    }
    plot_rmse_histogram(rmse_dict, os.path.join(PLOTS_DIR, 'rmse_histogram.png'))

    # 2. Forecasting metrics comparison
    print("   - Generating forecasting metrics comparison...")
    for metric in ['rmse_mean', 'mae_mean', 'mape_mean']:
        plot_metric_comparison(
            forecasting_metrics, metric,
            f'Forecasting Comparison: {metric.replace("_", " ").title()}',
            os.path.join(PLOTS_DIR, f'forecast_{metric}.png')
        )

    # 3. Capacity planning metrics comparison
    print("   - Generating capacity planning comparison...")
    for metric in ['u_max_mean', 'u_max_max', 'f_over_mean']:
        title = f'Capacity Planning: {metric.replace("_", " ").title()}'
        plot_capacity_bars(
            capacity_metrics, metric, title,
            os.path.join(PLOTS_DIR, f'capacity_{metric}.png')
        )

    # 4. Time series examples for representative links
    print("   - Generating time series examples...")

    # Find links with different characteristics
    sarima_rmse = np.array(sarima_metrics['per_link']['rmse'])
    lstm_rmse = np.array(lstm_metrics['per_link']['rmse'])

    # Link where LSTM does better
    diff = sarima_rmse - lstm_rmse
    best_lstm_link = np.argmax(diff)

    # Link where SARIMA does better
    best_sarima_link = np.argmin(diff)

    # Median performance link
    median_idx = np.argsort(sarima_rmse)[len(sarima_rmse) // 2]

    example_links = [
        (best_lstm_link, 'lstm_better'),
        (best_sarima_link, 'sarima_better'),
        (median_idx, 'median_perf')
    ]

    time_indices = np.arange(T_eff)

    for link_idx, label in example_links:
        predictions_dict = {
            'SARIMA': Y_pred_sarima[:, link_idx],
            'LSTM': Y_pred_lstm[:, link_idx]
        }
        capacity_dict = {
            'SARIMA': capacities['SARIMA'][link_idx],
            'LSTM': capacities['LSTM'][link_idx],
            'Oracle': capacities['Oracle'][link_idx]
        }
        plot_timeseries_comparison(
            time_indices, Y_true[:, link_idx],
            predictions_dict, link_idx, capacity_dict,
            os.path.join(PLOTS_DIR, f'timeseries_link_{link_idx}_{label}.png')
        )


def main():
    """Run capacity planning evaluation."""
    print("=" * 50)
    print("Capacity Planning Evaluation")
    print("=" * 50)

    # Load and align predictions
    print("\n1. Loading predictions...")
    predictions = load_predictions()
    print(f"   - Effective test length: {predictions['T_eff']}")
    print(f"   - Number of links: {predictions['num_links']}")

    # Compute capacities
    print("\n2. Computing capacities...")
    percentile = CONFIG['capacity_percentile']
    margin = CONFIG['capacity_margin']
    capacities = compute_capacities(predictions, percentile, margin)

    print(f"   Using {percentile}th percentile with {margin}x margin")
    for model, cap in capacities.items():
        print(f"   - {model}: mean={cap.mean():.2f}, max={cap.max():.2f}")

    # Compute utilization metrics
    print("\n3. Computing utilization metrics...")
    capacity_metrics = compute_utilization_metrics(
        predictions['Y_true'], capacities
    )

    for model, metrics in capacity_metrics.items():
        print(f"\n   {model}:")
        print(f"     Mean U_max: {metrics['u_max_mean']:.4f}")
        print(f"     Max U_max:  {metrics['u_max_max']:.4f}")
        print(f"     Mean f_over: {metrics['f_over_mean']*100:.2f}%")
        print(f"     Links >110%: {metrics['links_over_110']}")

    # Save capacity metrics
    save_json(capacity_metrics, os.path.join(RESULTS_DIR, 'capacity_planning.json'))

    # Load forecasting metrics for comparison
    print("\n4. Loading forecasting metrics...")
    sarima_metrics = load_json(os.path.join(RESULTS_DIR, 'sarima_metrics.json'))
    lstm_metrics = load_json(os.path.join(RESULTS_DIR, 'lstm_metrics.json'))

    forecasting_metrics = {
        'SARIMA': sarima_metrics['aggregated'],
        'LSTM': lstm_metrics['aggregated']
    }

    # Generate plots
    print("\n5. Generating plots...")
    generate_plots(predictions, capacities, forecasting_metrics, capacity_metrics)
    print(f"   Saved to: {PLOTS_DIR}")

    # Print summary table
    print_summary_table(forecasting_metrics, capacity_metrics)

    # Save combined results
    combined_results = {
        'forecasting': forecasting_metrics,
        'capacity_planning': capacity_metrics,
        'config': {
            'capacity_percentile': percentile,
            'capacity_margin': margin,
            'window_size': CONFIG['window_size'],
            'T_eff': predictions['T_eff'],
            'num_links': predictions['num_links']
        }
    }
    save_json(combined_results, os.path.join(RESULTS_DIR, 'combined_results.json'))

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

    return forecasting_metrics, capacity_metrics


if __name__ == '__main__':
    main()
