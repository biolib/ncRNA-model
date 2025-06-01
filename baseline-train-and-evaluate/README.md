# Pretraining and Fine-tuning Pipeline

This application provides an automated pipeline to pretrain a 1D Convolutional Neural Network (CNN) on RNA sequences and subsequently fine-tune and evaluate this model on a variety of downstream bioinformatics tasks. The core idea is to leverage large unlabeled RNA datasets to learn robust representations (pretraining) that can then be adapted to specific tasks with smaller, labeled datasets (fine-tuning).

## What it Does: The Workflow

The pipeline executes a series of automated steps to investigate the impact of pretraining data size on downstream task performance.

### 1. Input: What You Provide
*   **A FASTA File**: This is the primary input. It should contain a large collection of RNA sequences that will be used for the initial pretraining phase.

### 2. Preprocessing & Data Subsetting
*   **Sequence Counting**: The application first determines the total number of sequences in your input FASTA file.
*   **Logarithmic Subsetting**: To study the effect of pretraining data scale, the application generates several subsets of your input FASTA file. These subsets are created with logarithmically increasing numbers of sequences (e.g., 10, 100, 1000, ... up to the total number of sequences in your input file). Each subset is saved as a new FASTA file.

### 3. Iterative Pretraining
For **each** data subset created in the previous step:
*   **Model Pretraining**: A 1D CNN model is trained from scratch (or with a specified architecture) using a Masked Language Modeling (MLM) objective. This means the model learns to predict randomly masked nucleotides within the RNA sequences of the current subset.
    *   **Input**: The FASTA file for the current data subset.
    *   **Process**: The `pretrain_mlm_1dcnn.py` script is invoked. It uses an efficient iterable dataset loader suitable for large sequence files.
    *   **Output**: A pretrained model file (e.g., `final_model.pth`) specific to this data subset, along with training logs and metric plots.

### 4. Iterative Fine-tuning & Evaluation
For **each** pretrained model generated in the step above (i.e., one model per data subset):
*   The application takes this pretrained model and fine-tunes it on a predefined suite of downstream RNA analysis tasks. This allows us to see how a model pretrained on `X` sequences performs on task `Y`.
*   **Supported Downstream Tasks**:
    *   Splice Site Prediction (Acceptor & Donor)
    *   ncRNA Family Classification (with different noise levels)
    *   RNA Secondary Structure Prediction
    *   RNA Modification Site Prediction
*   **Fine-tuning Process**: For each downstream task:
    *   **Input**: The pretrained model (from step 3) and the specific task\'s labeled training, validation, and test datasets.
    *   **Process**: The `finetune_mlm_1dcnn.py` script is invoked. It loads the backbone of the pretrained model, adds a new task-specific classification head, and trains this composite model. The backbone can be initially frozen and then unfrozen during training.
    *   **Special Handling for Multi-Test Splice Site Tasks**: For splice site prediction (acceptor and donor), the model is fine-tuned once using the primary training/validation data and evaluated on a first test file. Then, for additional organism-specific test files (e.g., Danio, Fly, Thaliana, Worm), the *same fine-tuned model* is evaluated directly without retraining, using an `evaluate_only_on_test` mode.
    *   **Output**: For each task and each pretrained model:
        *   A fine-tuned model file (e.g., `finetuned_model.pt`).
        *   A CSV file containing detailed evaluation metrics (loss, accuracy, F1-score, precision, recall) on the test set(s) (e.g., `finetune_results.csv` or `finetune_results_TestOrganism.csv`).
        *   Logs of the fine-tuning process.

### 5. Output: What You Get
The application generates a well-organized directory structure containing all artifacts from the pipeline. For each initial data subset size, you will find:
*   The subset FASTA file itself.
*   The pretrained model trained on that subset.
*   Logs and metric plots from the pretraining phase.
*   For each downstream task:
    *   The fine-tuned model.
    *   Detailed evaluation results (CSV).
    *   Logs from the fine-tuning and evaluation phase.
    *   A plot of the performance.

This comprehensive output allows for detailed analysis of how the amount of pretraining data influences performance across various RNA-related predictive tasks.