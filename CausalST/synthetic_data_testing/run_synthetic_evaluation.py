import os
import subprocess

def run_workflow(dataset_number):
    """Run the complete synthetic data evaluation workflow"""
    
    print(f"\nProcessing dataset {dataset_number}")
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Train RF model
    print("\nTraining RF model...")
    rf_script = os.path.join(script_dir, 'rf_predictions.py')
    subprocess.run(['python', rf_script, '--dataset', str(dataset_number)])
    
    # Step 2: Train TARNet model
    print("\nTraining TARNet model...")
    tarnet_script = os.path.join(script_dir, 'tarnet_predictions.py')
    subprocess.run(['python', tarnet_script, '--dataset', str(dataset_number)])
    
    # Step 3: Evaluate RF predictions
    print("\nEvaluating RF predictions...")
    rf_eval_script = os.path.join(script_dir, 'evaluate_synthetic_predictions.py')
    subprocess.run(['python', rf_eval_script, '--dataset', str(dataset_number)])
    
    # Step 4: Evaluate TARNet predictions
    print("\nEvaluating TARNet predictions...")
    tarnet_eval_script = os.path.join(script_dir, 'evaluate_tarnet_predictions.py')
    subprocess.run(['python', tarnet_eval_script, '--dataset', str(dataset_number)])

def main():
    # List of dataset numbers to process
    datasets = [0]  # Add more numbers as needed
    
    for dataset in datasets:
        run_workflow(dataset)

if __name__ == "__main__":
    main() 