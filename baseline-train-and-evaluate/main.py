import argparse
import subprocess
import os
os.environ["WANDB_SILENT"] = "true"
import math
import shutil
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random  # Add this import
from dataset.dataset_preprocessing import split_sequences # Added import

def count_sequences_fasta(fasta_path):
    """Counts the number of sequences in a FASTA file."""
    count = 0
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def create_subset_fasta(original_fasta_path, num_sequences, output_fasta_path, random_seed=None):
    """Creates a random subset FASTA file from the original."""
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        
    # First count the total number of sequences
    total_sequences = count_sequences_fasta(original_fasta_path)
    
    if total_sequences <= num_sequences:
        # If we want all sequences or more than exists, just copy the file
        shutil.copy(original_fasta_path, output_fasta_path)
        print(f"Created subset FASTA: {output_fasta_path} with all {total_sequences} sequences (requested {num_sequences}).")
        return
    
    # Randomly select sequence indices to include
    selected_indices = sorted(random.sample(range(total_sequences), num_sequences))
    
    # Read through the file and extract only the selected sequences
    current_seq_idx = -1
    in_selected_sequence = False
    actual_sequences_written = 0
    
    with open(original_fasta_path, 'r') as infile, open(output_fasta_path, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                current_seq_idx += 1
                in_selected_sequence = current_seq_idx in selected_indices
                if in_selected_sequence:
                    actual_sequences_written += 1
            
            if in_selected_sequence:
                outfile.write(line)
    
    print(f"Created random subset FASTA: {output_fasta_path} with {actual_sequences_written} sequences.")

def load_yaml_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml_config(config_data, config_path):
    """Saves data to a YAML configuration file."""
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run pretraining and fine-tuning pipelines for different dataset sizes and model architectures.")
    parser.add_argument("--model_architecture", "-a", type=str, default="cnn", choices=["cnn", "transformer"], help="Model architecture to use (cnn or transformer).")
    parser.add_argument("--input_fasta", "-i", type=str, required=True, help="Path to the full input FASTA file for pretraining.")
    parser.add_argument("--output_base_dir", type=str, default="output", help="Base directory to save all outputs.")
    parser.add_argument("--log_wandb", "-w",action="store_true", help="Whether to log to WANDB.")
    parser.add_argument("--wandb_config", "-g", type=str, default="configs/general.yml", help="WANDB config file.")
    
    # Pretraining related arguments
    parser.add_argument("--pretrain_dataset_config_template", "-d", type=str, default="configs/data/basic.yml", help="Path to the pretraining dataset YAML config template.")
    parser.add_argument("--pretrain_only", "-p", action="store_true", help="Whether to only run pretraining and skip fine-tuning.")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to the pretrained model to use for fine-tuning.")
    
    # Global Fine-tuning Hyperparameters (applied to all tasks)
    parser.add_argument("--finetune_only", "-f", action="store_true", help="Whether to only run fine-tuning and skip pretraining.")
    parser.add_argument("--finetune_max_seq_len", type=int, default=512, help="Max sequence length for fine-tuning.")
    parser.add_argument("--finetune_learning_rate", type=float, default=0.001, help="Learning rate for fine-tuning.")
    parser.add_argument("--finetune_batch_size", type=int, default=64, help="Batch size for fine-tuning.")
    parser.add_argument("--finetune_num_epochs", type=int, default=5, help="Number of epochs for fine-tuning.")
    parser.add_argument("--finetune_unfreeze_epoch", type=int, default=-1, help="Epoch to unfreeze backbone during fine-tuning. -1 to keep frozen.")

    # --- Arguments for specific fine-tuning task data paths ---
    # Splice Site Acceptor
    parser.add_argument("--ft_splice_acceptor_train_data", type=str, default="finetune_data/splice_site_prediction/acceptor/train.tsv")
    parser.add_argument("--ft_splice_acceptor_val_data", type=str, default="finetune_data/splice_site_prediction/acceptor/valid.tsv")

    # Splice Site Donor
    parser.add_argument("--ft_splice_donor_train_data", type=str, default="finetune_data/splice_site_prediction/donor/train.tsv")
    parser.add_argument("--ft_splice_donor_val_data", type=str, default="finetune_data/splice_site_prediction/donor/valid.tsv")

    # ncRNA Family bnoise0
    parser.add_argument("--ft_ncrna_b0_train_data", type=str, default="finetune_data/ncrna_family_classification/bnoise0/train.tsv")
    parser.add_argument("--ft_ncrna_b0_val_data", type=str, default="finetune_data/ncrna_family_classification/bnoise0/valid.tsv")
    parser.add_argument("--ft_ncrna_b0_test_data", type=str, default="finetune_data/ncrna_family_classification/bnoise0/test.tsv")

    # ncRNA Family bnoise200
    parser.add_argument("--ft_ncrna_b200_train_data", type=str, default="finetune_data/ncrna_family_classification/bnoise200/train.tsv")
    parser.add_argument("--ft_ncrna_b200_val_data", type=str, default="finetune_data/ncrna_family_classification/bnoise200/valid.tsv")
    parser.add_argument("--ft_ncrna_b200_test_data", type=str, default="finetune_data/ncrna_family_classification/bnoise200/test.tsv")

    # Secondary Structure (BP RNA SPOT)
    parser.add_argument("--ft_secstruct_train_data", type=str, default="finetune_data/secondary_structure/bprna_spot/train.csv")
    parser.add_argument("--ft_secstruct_val_data", type=str, default="finetune_data/secondary_structure/bprna_spot/validation.csv")
    parser.add_argument("--ft_secstruct_test_data", type=str, default="finetune_data/secondary_structure/bprna_spot/test.csv")

    # Modification Site
    parser.add_argument("--ft_modsite_train_data", type=str, default="finetune_data/modification_site_prediction/train.tsv")
    parser.add_argument("--ft_modsite_val_data", type=str, default="finetune_data/modification_site_prediction/valid.tsv")
    parser.add_argument("--ft_modsite_test_data", type=str, default="finetune_data/modification_site_prediction/test.tsv")

    parser.add_argument("--python_executable", type=str, default="python3", help="Python executable to use (e.g., python3, python).")
    
    parser.add_argument("--start_subset_size", type=int, default=10, help="Starting size for the subset of sequences to use for pretraining.")

    parser.add_argument("--cnn_model_config_template", "-c", type=str, default="configs/model/1d_cnn.yml", help="Path to the CNN model config template.")
    parser.add_argument("--transformer_model_config_template", "-t", type=str, default="configs/model/transformer.yml", help="Path to the Transformer model config template.")

    args = parser.parse_args()

    output_base_dir = Path(args.output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    all_finetune_results = [] # Initialize list to store all results for plotting

    # Determine pretrain script and model config based on architecture
    if args.model_architecture == "cnn":
        pretrain_script_name = "pretrain_mlm_1dcnn.py"
        finetune_script_name = "finetune_mlm_1dcnn.py"
        pretrain_model_config_template_path = args.cnn_model_config_template
    elif args.model_architecture == "transformer":
        pretrain_script_name = "pretrain_mlm_transformer.py"
        finetune_script_name = "finetune_mlm_transformer.py"
        pretrain_model_config_template_path = args.transformer_model_config_template
    else:
        print(f"Error: Unsupported model architecture '{args.model_architecture}'. Exiting.")
        return
    if args.pretrain_only:
        logger.warning("MODE: Pretraining only. This will not run any fine-tuning tasks.")
        logger.info(f"Running pretraining only for {args.input_fasta} with {args.model_architecture} model.")
    else:
        logger.info(f"Running pretraining and fine-tuning for {args.input_fasta} with {args.model_architecture} model.")
    
    logger.info(f"Using model architecture: {args.model_architecture}")
    logger.info(f"Pretrain script: {pretrain_script_name}, Finetune script: {finetune_script_name}")
    logger.info(f"Pretrain model config template: {pretrain_model_config_template_path}")

    pretrain_model_config_template_data = load_yaml_config(pretrain_model_config_template_path)
    # Load pretrain_dataset_config_template_data to get pretrain_max_length and other params
    pretrain_dataset_config_template_for_params = load_yaml_config(args.pretrain_dataset_config_template)
    pretrain_max_length = pretrain_dataset_config_template_for_params['data'].get('max_length', 1024)
    # Get stride and window_size from dataset config for preprocessing, or use defaults in split_sequences
    pretrain_stride = pretrain_dataset_config_template_for_params['data'].get('stride', None) 
    pretrain_window_size = pretrain_dataset_config_template_for_params['data'].get('window_size', None)
    # Get random seed for reproducible sequence selection if specified in config
    random_seed = pretrain_dataset_config_template_for_params['data'].get('seed', 42)
    print(f"Using random seed from data config: {random_seed}")

    # For Transformer, these might be named differently or not all applicable
    # Ensure these are present in both cnn.yml and transformer.yml or handle missing keys
    if args.model_architecture == "cnn":
        ft_embedding_dim = pretrain_model_config_template_data['model']['embedding_dim']
        ft_hidden_dim = pretrain_model_config_template_data['model']['hidden_dim']
        ft_num_layers = pretrain_model_config_template_data['model']['num_layers']
        ft_kernel_size = pretrain_model_config_template_data['model']['kernel_size']
        ft_dropout_rate = pretrain_model_config_template_data['model']['dropout_rate']
    elif args.model_architecture == "transformer":
        ft_embedding_dim = pretrain_model_config_template_data['model']['embedding_dim']
        ft_nhead = pretrain_model_config_template_data['model']['num_heads']
        ft_num_encoder_layers = pretrain_model_config_template_data['model']['num_layers']
        ft_dim_feedforward = pretrain_model_config_template_data['model']['dim_feedforward']
        ft_dropout_rate = pretrain_model_config_template_data['model']['dropout_rate']

    total_sequences = count_sequences_fasta(args.input_fasta)
    if total_sequences == 0:
        print(f"Error: No sequences found in {args.input_fasta}. Exiting.")
        return

    subset_sizes = []
    current_size = int(args.start_subset_size)
    while current_size < total_sequences:
        subset_sizes.append(current_size)
        current_size *= 10
    subset_sizes.append(total_sequences) 
    subset_sizes = sorted(list(set(s for s in subset_sizes if s <= total_sequences)))

    print("Using fasta file: ", args.input_fasta)
    print(f"Will run experiments for sequence counts: {subset_sizes}")

    # --- Define Fine-tuning Task Configurations ---
    splice_site_test_files = ["test_Danio.tsv", "test_Fly.tsv", "test_Thaliana.tsv", "test_Worm.tsv"]
    
    finetuning_tasks = [
        {
            "name": "splice_site_acceptor", "script_task_name": "splice_site",
            "train_data": args.ft_splice_acceptor_train_data, "val_data": args.ft_splice_acceptor_val_data, 
            "test_files": splice_site_test_files,
            "test_data_base_dir": str(Path(args.ft_splice_acceptor_train_data).parent)
        },
        {
            "name": "splice_site_donor", "script_task_name": "splice_site",
            "train_data": args.ft_splice_donor_train_data, "val_data": args.ft_splice_donor_val_data, 
            "test_files": splice_site_test_files,
            "test_data_base_dir": str(Path(args.ft_splice_donor_train_data).parent)
        },
        {
            "name": "ncrna_family_bnoise0", "script_task_name": "ncrna_family",
            "train_data": args.ft_ncrna_b0_train_data, "val_data": args.ft_ncrna_b0_val_data, "test_data": args.ft_ncrna_b0_test_data
        },
        {
            "name": "ncrna_family_bnoise200", "script_task_name": "ncrna_family",
            "train_data": args.ft_ncrna_b200_train_data, "val_data": args.ft_ncrna_b200_val_data, "test_data": args.ft_ncrna_b200_test_data
        },
        {
            "name": "secondary_structure_bprna_spot", "script_task_name": "secondary_structure",
            "train_data": args.ft_secstruct_train_data, "val_data": args.ft_secstruct_val_data, "test_data": args.ft_secstruct_test_data
        },
        {
            "name": "modification_site", "script_task_name": "modification_site",
            "train_data": args.ft_modsite_train_data, "val_data": args.ft_modsite_val_data, "test_data": args.ft_modsite_test_data
        },
    ]

    for num_seq in subset_sizes:
        print(f"--- Processing for {num_seq} sequences using {args.model_architecture} ---")
        
        current_run_dir = output_base_dir / f"{args.model_architecture}_{num_seq}_sequences"
        current_run_dir.mkdir(parents=True, exist_ok=True)

        pretrain_run_dir = current_run_dir / "pretrain"
        pretrain_run_dir.mkdir(parents=True, exist_ok=True)

        # --- Create subset FASTA for this run ---
        subset_fasta_path = pretrain_run_dir / os.path.basename(args.input_fasta).replace(".fasta", f"_subset_{num_seq}.fasta")
        create_subset_fasta(args.input_fasta, num_seq, subset_fasta_path, random_seed)
        
        current_pretrain_dataset_config_data = load_yaml_config(args.pretrain_dataset_config_template) # Load fresh template
        temp_dataset_config_path = pretrain_run_dir / "temp_dataset_config.yml"
        current_pretrain_dataset_config_data['data']['dataset_filename'] = str(subset_fasta_path.resolve())
        save_yaml_config(current_pretrain_dataset_config_data, temp_dataset_config_path)

        pretrain_model_config_data = load_yaml_config(pretrain_model_config_template_path)
        if 'training' not in pretrain_model_config_data: pretrain_model_config_data['training'] = {}
        pretrain_model_config_data['training']['model_save_path'] = str(pretrain_run_dir.resolve())
        temp_model_config_path = pretrain_run_dir / "temp_model_config.yml"
        save_yaml_config(pretrain_model_config_data, temp_model_config_path)

        if args.finetune_only:
            print(f"Skipping pretraining, loading pretrained model from {args.pretrained_model_path}.")
            pretrained_model_for_finetune = Path(args.pretrained_model_path)
        else:
            import randomname
            import random
            
            group_name = randomname.get_name() + str(random.randint(100000, 999999))
            print(f"Using wandb group name: {group_name}")
            
            # 2. Run Pretraining
            print(f"Starting pretraining for {num_seq} sequences with {args.model_architecture}...")
            cmd_pretrain = [
                args.python_executable, pretrain_script_name, # Use dynamic script name
                "--model_config", str(temp_model_config_path.resolve()),
                "--dataset_config", str(temp_dataset_config_path.resolve()),
                "-g", args.wandb_config,
                "--wandb_group", group_name,
            ]
            if args.log_wandb:
                cmd_pretrain.append("-w")
            
            pretrain_stdout_log_path = pretrain_run_dir / f"pretrain_run_{args.model_architecture}_stdout.log"
            try:
                with open(pretrain_stdout_log_path, 'w') as plog: # File still created, but subprocess won't pipe to it.
                    # Removed stdout=plog, stderr=subprocess.STDOUT to allow terminal output for tqdm
                    subprocess.run(cmd_pretrain, check=True)
                
                # Move the hardcoded pretrain log (e.g., mlm_1dcnn.log or mlm_transformer.log)
                hardcoded_pretrain_log_name = "mlm_1dcnn.log" if args.model_architecture == "cnn" else "mlm_transformer.log"
                hardcoded_pretrain_log = Path(hardcoded_pretrain_log_name) 
                if hardcoded_pretrain_log.exists():
                    shutil.move(str(hardcoded_pretrain_log), str(pretrain_run_dir / f"pretrain_script_internal_{args.model_architecture}.log"))
                print(f"Pretraining finished. Outputs in {pretrain_run_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error during pretraining for {num_seq} sequences. Check logs in {pretrain_run_dir}.")
                print(f"Command was: {' '.join(e.cmd)}")
                print(f"Return code: {e.returncode}")
                continue # Skip to next subset size
            
            if args.pretrain_only:
                exit(0)

            # 3. Prepare for Fine-tuning
            pretrained_model_for_finetune = pretrain_run_dir / "best_model.pth"
        if not pretrained_model_for_finetune.exists():
            print(f"Pretrained model {pretrained_model_for_finetune} not found. Skipping fine-tuning for {num_seq} sequences.")
            continue
        
        # --- Loop through each fine-tuning task ---
        for task_config in finetuning_tasks:
            task_name = task_config["name"]
            script_task_name = task_config["script_task_name"]
            
            finetune_task_run_dir = current_run_dir / f"finetune_{task_name}"
            finetune_task_run_dir.mkdir(parents=True, exist_ok=True)
            
            finetuned_model_save_path = finetune_task_run_dir / "finetuned_model.pt"

            # Determine test paths: single or multiple
            if "test_files" in task_config: # For splice site tasks with multiple test files
                current_test_paths_info = [
                    {"path": Path(task_config["test_data_base_dir"]) / test_file, "stem": Path(test_file).stem}
                    for test_file in task_config["test_files"]
                ]
            else: # For other tasks with a single test file
                current_test_paths_info = [{"path": Path(task_config["test_data"]), "stem": None}]


            for idx, test_info in enumerate(current_test_paths_info): # Added enumerate
                current_test_path = test_info["path"]
                test_file_name_stem = test_info["stem"]

                print(f"Starting fine-tuning for task '{task_name}' (Test: {current_test_path.name}) using {args.model_architecture} model pretrained on {num_seq} sequences...")
                
                cmd_finetune_base = [
                    args.python_executable, finetune_script_name, # Use dynamic script name
                    "--pretrained_model_path", str(pretrained_model_for_finetune.resolve()),
                    "--train_path", task_config["train_data"],
                    "--val_path", task_config["val_data"],
                    "--test_path", str(current_test_path.resolve()), 
                    "--model_save_path", str(finetuned_model_save_path.resolve()), 
                    "--task", script_task_name,
                    "--task_name", task_name,
                    "-g", args.wandb_config,
                    "--wandb_group", group_name,
                    # Model specific args for finetuning script
                ]
                if args.log_wandb:
                    cmd_finetune_base.append("-w")

                # Add model-specific architecture parameters for finetuning script
                if args.model_architecture == "cnn":
                    cmd_finetune_model_params = [
                        "--embedding_dim", str(ft_embedding_dim),
                        "--hidden_dim", str(ft_hidden_dim),
                        "--num_layers", str(ft_num_layers),
                        "--kernel_size", str(ft_kernel_size),
                        "--dropout_rate", str(ft_dropout_rate),
                    ]
                elif args.model_architecture == "transformer":
                    cmd_finetune_model_params = [
                        "--embedding_dim", str(ft_embedding_dim),
                        "--nhead", str(ft_nhead),
                        "--num_encoder_layers", str(ft_num_encoder_layers),
                        "--dim_feedforward", str(ft_dim_feedforward),
                        "--dropout_rate", str(ft_dropout_rate),
                    ]
                
                cmd_finetune_hyperparams = [
                    "--max_seq_len", str(args.finetune_max_seq_len),
                    "--learning_rate", str(args.finetune_learning_rate),
                    "--batch_size", str(args.finetune_batch_size),
                    "--num_epochs", str(args.finetune_num_epochs),
                    "--unfreeze_epoch", str(args.finetune_unfreeze_epoch),
                ]

                cmd_finetune = cmd_finetune_base + cmd_finetune_model_params + cmd_finetune_hyperparams

                # If this is a multi-test task (like splice sites) AND it's not the first test file,
                # then add the evaluate_only_on_test flag.
                if "test_files" in task_config and idx > 0:
                    cmd_finetune.append("--evaluate_only_on_test")
                    print("   (Running in evaluation-only mode for this test file)")
                
                # Adjust log and result file names for multiple tests
                if test_file_name_stem:
                    finetune_stdout_log_path = finetune_task_run_dir / f"finetune_run_stdout_{test_file_name_stem}.log"
                    final_results_csv_name = f"finetune_results_{test_file_name_stem}.csv"
                else:
                    finetune_stdout_log_path = finetune_task_run_dir / "finetune_run_stdout.log"
                    final_results_csv_name = "finetune_results.csv"

                try:
                    with open(finetune_stdout_log_path, 'w') as flog: # File still created, but subprocess won't pipe to it.
                        # Removed stdout=flog, stderr=subprocess.STDOUT to allow terminal output for tqdm
                        subprocess.run(cmd_finetune, check=True)

                    # Move the results CSV
                    finetune_results_csv_in_cwd = Path("finetune_results.csv") # Script always outputs this name
                    destination_results_csv_path = finetune_task_run_dir / final_results_csv_name

                    if finetune_results_csv_in_cwd.exists():
                        shutil.move(str(finetune_results_csv_in_cwd), str(destination_results_csv_path))
                        #print(f"Moved finetune_results.csv to {destination_results_csv_path}")
                    elif destination_results_csv_path.exists():
                         print(f"{destination_results_csv_path.name} already in {finetune_task_run_dir} (possibly moved by script or previous run).")
                    else:
                        print(f"Warning: finetune_results.csv not found in CWD. Check fine-tune script's output behavior and logs at {finetune_stdout_log_path}.")
                    
                    # Collect results for plotting
                    if destination_results_csv_path.exists():
                        try:
                            results_data_df = pd.read_csv(destination_results_csv_path)
                            if not results_data_df.empty:
                                test_id_for_plot = test_info["stem"] if test_info["stem"] else "default"
                                row = results_data_df.iloc[0].to_dict()
                                row["num_sequences"] = num_seq
                                row["task_name"] = task_name
                                row["test_identifier"] = test_id_for_plot
                                all_finetune_results.append(row)
                            else:
                                print(f"Warning: Results CSV {destination_results_csv_path} is empty for task {task_name}, test {test_id_for_plot}, num_seq {num_seq}.")
                        except Exception as e:
                            print(f"Warning: Could not read or parse {destination_results_csv_path} for task {task_name}, test {test_id_for_plot}, num_seq {num_seq}. Error: {e}")
                    else:
                        print(f"Warning: Results CSV {destination_results_csv_path} not found after fine-tuning for task {task_name}, test {test_id_for_plot}, num_seq {num_seq}.")

                    print(f"Fine-tuning for task '{task_name}' (Test: {current_test_path.name}) finished. Outputs in {finetune_task_run_dir}")

                except subprocess.CalledProcessError as e:
                    print(f"Error during fine-tuning for task '{task_name}' (Test: {current_test_path.name}) with model from {num_seq} sequences. Check logs in {finetune_stdout_log_path}.")
                    print(f"Command was: {' '.join(e.cmd)}")
                    print(f"Return code: {e.returncode}")
                    # Continue to next test file or task even if this one fails
                
                # If it's not a multi-test task (i.e. "test_files" not in task_config), this inner loop runs once.
                # No explicit break needed here as current_test_paths_info will have only one element.
            
        print(f"--- Completed all fine-tuning tasks for {num_seq} sequences ---")

    print("All experiments finished. Generating performance plots...")

    # --- Plotting ---
    if not all_finetune_results:
        print("No fine-tuning results collected. Skipping plotting.")
    else:
        results_df = pd.DataFrame(all_finetune_results)
        unique_plot_tasks = results_df["task_name"].unique()

        for current_plot_task_name in unique_plot_tasks:
            task_df = results_df[results_df["task_name"] == current_plot_task_name]
            if task_df.empty:
                continue

            # Special handling for modification_site: plot per-label AUROC
            if current_plot_task_name == "modification_site":
                current_plot_task_name = current_plot_task_name.replace("_", " ")
                # Find all AUROC columns
                auroc_cols = [col for col in task_df.columns if col.startswith("auroc_label_") and not col.startswith("auroc_labelname_")]
                # Use label names as in the paper
                mod_labels = ['Am', 'Cm', 'Gm', 'Tm', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Φ', 'I']
                if auroc_cols:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    # Use a wide color palette (tab20)
                    cmap = plt.cm.get_cmap('tab20', len(auroc_cols))
                    for i, col in enumerate(auroc_cols):
                        label_idx = int(col.split('_')[-1])
                        label_name = mod_labels[label_idx] if label_idx < len(mod_labels) else f'Label {label_idx}'
                        color = cmap(i)
                        ax.plot(task_df["num_sequences"], task_df[col], marker='o', linestyle='-', label=label_name, color=color)
                    # Also plot average AUROC
                    if "auroc_avg" in task_df.columns:
                        ax.plot(task_df["num_sequences"], task_df["auroc_avg"], marker='x', linestyle='--', color='black', label='Average AUROC')
                    ax.set_xlabel("Number of pretraining sequences (log scale)")
                    ax.set_xscale('log')
                    ax.set_ylabel("AUROC")
                    ax.set_title(f"Per-label AUROC vs. pretraining size for {current_plot_task_name}", pad=20)
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
                    plt.grid(True, which="both", ls="--")
                    fig.tight_layout()
                    plot_filename = output_base_dir / f"plot_{args.model_architecture}_{current_plot_task_name}_per_label_auroc_vs_pretrain_size.png"
                    plt.savefig(plot_filename, bbox_inches='tight')
                    print(f"Saved per-label AUROC plot to {plot_filename}")
                    plt.close(fig)
            else:
                # Determine if this is a splice site task (acceptor or donor)
                is_splice_site = current_plot_task_name.startswith("splice_site_")
                if is_splice_site:
                    current_plot_task_name = current_plot_task_name.replace("_", " ")
                    fig, ax = plt.subplots(figsize=(14, 8))
                    # Assign a unique color to each test_id using a colormap
                    sorted_test_identifiers = sorted(task_df["test_identifier"].unique()) if "test_identifier" in task_df.columns else [None]
                    cmap = plt.cm.get_cmap('tab10', len(sorted_test_identifiers))
                    ax.set_xlabel("Number of pretraining sequences (log scale)")
                    ax.set_xscale('log')
                    ax.set_ylabel("F1 score", color='black')
                    ax.tick_params(axis='y', labelcolor='black')
                    lines = []
                    labels = []
                    for i, test_id in enumerate(sorted_test_identifiers):
                        test_id_label = test_id.split('test_')[1] if test_id is not None and test_id.startswith('test_') else (test_id if test_id is not None else "")
                        subset_df = task_df[task_df["test_identifier"] == test_id].sort_values("num_sequences") if test_id is not None else task_df.sort_values("num_sequences")
                        if subset_df.empty:
                            continue
                        color = cmap(i)
                        # Plot only F1 Macro
                        line = ax.plot(subset_df["num_sequences"], subset_df["f1_macro"], marker='x', linestyle='-', label=f'{test_id_label} F1 score' if test_id_label else 'F1 score', color=color)
                        lines.extend(line)
                        labels.append(f'{test_id_label} F1 score' if test_id_label else 'F1 score')
                    plt.title(f"Performance vs. pretraining size for {current_plot_task_name}", pad=20)
                    if lines:
                        ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
                    plt.grid(True, which="both", ls="--")
                    fig.tight_layout()
                    plot_filename = output_base_dir / f"plot_{args.model_architecture}_{current_plot_task_name}_performance_vs_pretrain_size.png"
                    plt.savefig(plot_filename, bbox_inches='tight')
                    print(f"Saved performance plot to {plot_filename}")
                    plt.close(fig)
                else:
                    # ncRNA family and secondary structure: plot both accuracy and F1 Macro
                    current_plot_task_name = current_plot_task_name.replace("_", " ")
                    fig, ax1 = plt.subplots(figsize=(14, 8))
                    color_accuracy = 'tab:blue'
                    color_f1 = 'tab:red'
                    ax1.set_xlabel("Number of pretraining sequences (log scale)")
                    ax1.set_xscale('log')
                    ax1.set_ylabel("Accuracy", color=color_accuracy)
                    ax1.tick_params(axis='y', labelcolor=color_accuracy)
                    ax2 = ax1.twinx()
                    ax2.set_ylabel("F1 score", color=color_f1)
                    ax2.tick_params(axis='y', labelcolor=color_f1)
                    lines = []
                    labels = []
                    sorted_test_identifiers = sorted(task_df["test_identifier"].unique()) if "test_identifier" in task_df.columns else [None]
                    for test_id in sorted_test_identifiers:
                        subset_df = task_df[task_df["test_identifier"] == test_id].sort_values("num_sequences") if test_id is not None else task_df.sort_values("num_sequences")
                        if subset_df.empty:
                            continue
                        # Plot Accuracy
                        line1 = ax1.plot(subset_df["num_sequences"], subset_df["test_accuracy"], marker='o', linestyle='-', label=f'Accuracy', color=color_accuracy)
                        lines.extend(line1)
                        labels.append(f'Accuracy')
                        # Plot F1 Macro
                        line2 = ax2.plot(subset_df["num_sequences"], subset_df["f1_macro"], marker='x', linestyle='--', label=f'F1 score', color=color_f1)
                        lines.extend(line2)
                        labels.append(f'F1 score')
                    plt.title(f"Performance vs. pretraining size for {current_plot_task_name}", pad=20)
                    if lines:
                        ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
                    plt.grid(True, which="both", ls="--")
                    fig.tight_layout()
                    plot_filename = output_base_dir / f"plot_{args.model_architecture}_{current_plot_task_name}_performance_vs_pretrain_size.png"
                    plt.savefig(plot_filename, bbox_inches='tight')
                    print(f"Saved performance plot to {plot_filename}")
                    plt.close(fig)

    print("Plotting complete. All processes finished.")

# Add logger setup at the beginning of the script if not already present
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    main()