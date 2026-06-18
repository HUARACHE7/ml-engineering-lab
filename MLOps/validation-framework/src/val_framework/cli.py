import argparse
import os
from experiment_tracker.tracker import MLTracker

def display_leaderboard(db_path: str, experiment_name: str, sort_by_metric: str, ascending: bool = False):
    """
    Reads data from the tracker database and displays a formatted leaderboard in the console.
    """
    if not os.path.exists(db_path):
        print(f"Error: Database not found at path: {db_path}")
        return

    tracker = MLTracker(experiment_name=experiment_name, db_path=db_path)
    
    runs = tracker.db.get_experiment_runs_summary(experiment_name)
    
    if not runs:
        print(f"No saved runs found for experiment '{experiment_name}'.")
        return

    metric_key = f"m:{sort_by_metric}"
    
    valid_runs = [r for r in runs if r.get("Status") == "FINISHED" and metric_key in r]
    
    if not valid_runs:
        print(f"No successful runs found with metric '{sort_by_metric}'.")
        print("Available metrics in the database:")
        available_metrics = set()
        for r in runs:
            for k in r.keys():
                if k.startswith("m:"):
                    available_metrics.add(k.replace("m:", ""))
        print(", ".join(available_metrics) if available_metrics else "No metrics available")
        return

    valid_runs.sort(key=lambda x: float(x[metric_key]), reverse=not ascending)

    print("\n" + "="*80)
    print(f" LEADERBOARD FOR EXPERIMENT: {experiment_name} (Sorted by {sort_by_metric})")
    print("="*80)
    print(f"{'ID':<5} | {'Run Name':<30} | {sort_by_metric:<12} | {'Parameters':<25}")
    print("-"*80)

    for run in valid_runs:
        run_id = run.get("id", run.get("run_id", "N/A"))
        name = run.get("Name", "N/A")
        metric_val = f"{float(run[metric_key]):.4f}"
        
        params = [f"{k.replace('p:', '')}={v}" for k, v in run.items() if k.startswith("p:")]
        params_str = ", ".join(params) if params else "none"
        
        if len(name) > 28:
            name = name[:25] + "..."
            
        print(f"{run_id:<5} | {name:<30} | {metric_val:<12} | {params_str:<25}")
        
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CLI for viewing model validation results.")
    
    parser.add_argument(
        "--db", 
        type=str, 
        default="validation.db", 
        help="Path to the SQLite tracker database file (default: validation.db)"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        help="Name of the experiment to analyze"
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default="R2_score", 
        help="Metric to rank models by (default: R2_score)"
    )
    parser.add_argument(
        "--asc", 
        action="store_true", 
        help="Sort in ascending order (useful for error metrics like MSE or MAE)"
    )

    args = parser.parse_args()
    
    display_leaderboard(
        db_path=args.db, 
        experiment_name=args.experiment, 
        sort_by_metric=args.metric, 
        ascending=args.asc
    )

if __name__ == "__main__":
    main()