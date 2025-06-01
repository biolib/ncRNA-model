import torch.nn as nn
import torch

import os
os.environ["WANDB_SILENT"] = "true"
import yaml
import argparse
import math

import logging
from tqdm import tqdm

from dataset.iterable_dataset import MaskedRNAIterableDataset, NUCLEOTIDE_VOCAB, CharTokenizer
from dataset.dataset_preprocessing import split_sequences

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('mlm_1dcnn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', '-m',type=str, default=f'{os.path.dirname(__file__)}/configs/model/1d_cnn.yml', help='Path to model YAML config file')
parser.add_argument('--dataset_config', '-d', type=str, default=f'{os.path.dirname(__file__)}/configs/data/basic.yml', help='Path to dataset YAML config file')
parser.add_argument('--general_config', '-g', type=str, default=f'{os.path.dirname(__file__)}/configs/general.yml', help='Path to general YAML config file')
parser.add_argument('--wandb', '-w', action='store_true', help='Enable wandb logging')
parser.add_argument('--wandb_group', type=str, default='group_name', help='Wandb group name.')

class MLM1DConv(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, kernel_size, dropout_rate, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        conv_layers = []
        in_channels = embedding_dim
        
        for _ in range(num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = hidden_dim
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Reshape for 1D convolution
        conv_input = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # Apply convolutions
        conv_output = self.conv_layers(conv_input)  # [batch, hidden_dim, seq_len]
        
        # Reshape back
        conv_output = conv_output.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        logits = self.output(conv_output)  # [batch, seq_len, vocab_size]
        return logits
    
def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, device, model_save_path, train_size, val_size, batch_size, log_wandb, early_stopping_patience=20, min_delta=1e-4):
    # Ignore -100 index in loss calculation
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    logger.info(f"Starting training for {num_epochs} epochs.")
    
    train_losses = []
    train_accuracies = []
    
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} of {num_epochs}")
        # Training
        model.train()
        
        train_loss = 0
        
        correct_preds = 0
        total_masked = 0
        
        for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, total=math.ceil(train_size/batch_size) if train_size is not None else None, unit="batch")):
            input_ids, labels, _ = data
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # Calculate loss only on masked tokens
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy on masked tokens
            masked_positions = (labels.view(-1) != -100)
            predictions = outputs.view(-1, outputs.size(-1)).argmax(dim=-1)
            correct = (predictions[masked_positions] == labels.view(-1)[masked_positions]).sum().item()
            correct_preds += correct
            total_masked += masked_positions.sum().item()
            
        avg_train_loss = train_loss / (i + 1)
        avg_train_acc = correct_preds / total_masked
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        running_vloss = 0.0
        vcorrect_preds = 0
        vtotal_masked = 0
        
        model.eval()
        

        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels, _ = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs.view(-1, voutputs.size(-1)), vlabels.view(-1))
                running_vloss += vloss.item()
                
                vmasked_positions = (vlabels.view(-1) != -100)
                vpredictions = voutputs.view(-1, voutputs.size(-1)).argmax(dim=-1)
                vcorrect = (vpredictions[vmasked_positions] == vlabels.view(-1)[vmasked_positions]).sum().item()
                vcorrect_preds += vcorrect
                vtotal_masked += vmasked_positions.sum().item()

        avg_val_acc = vcorrect_preds / vtotal_masked
        avg_val_loss = running_vloss / (i + 1)
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        logger.info(f'LOSS train {avg_train_loss:.4f} valid {avg_val_loss:.4f}')
        logger.info(f'ACC train {avg_train_acc:.4f} valid {avg_val_acc:.4f}')
        
        if log_wandb:
            import wandb
            wandb.log({
                'train/loss': avg_train_loss,
                'train/accuracy': avg_train_acc,
                'val/loss': avg_val_loss,
                'val/accuracy': avg_val_acc,
            }, step=epoch)
        
        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saving best model.")
            torch.save(model.state_dict(), f'{model_save_path}/best_model.pth')
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break


    logger.info("Training finished. Saving final model.")
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, model_save_path)
    
    # Write epoch metrics to CSV
    csv_path = os.path.join(model_save_path, 'training_metrics.csv')
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], train_accuracies[i], val_losses[i], val_accuracies[i]])
    logger.info(f"Training metrics CSV saved to {csv_path}")
    
    torch.save(model.state_dict(), f'{model_save_path}/1dcnn_model_acc_{avg_train_acc:.4f}_vacc_{avg_val_acc:.4f}_loss_{avg_train_loss:.4f}_vloss_{avg_val_loss:.4f}.pth')

def plot_metrics(losses, accuracies, val_losses, val_accuracies, output_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    plt.figure(figsize=(14, 6)) # Adjusted figure size for potentially two plots, or one wider one
    sns.set_style("whitegrid")

    plt.subplot(2, 2, 1)  # Changed to 2x2 grid, top-left
    plt.plot(range(1, len(losses) + 1), losses, label='Train Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)  # Changed to 2x2 grid, top-right
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Train Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)  # Changed to 2x2 grid, bottom-left
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)  # Changed to 2x2 grid, bottom-right
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_save_path = os.path.join(output_path, 'training_metrics.png')
    plt.savefig(plot_save_path)
    logger.info(f"Training metrics plot saved to {output_path}")
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args: argparse.Namespace):
    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)
    
    if args.wandb:
        import wandb
        # from dotenv import load_dotenv
        
        # load_dotenv()
        general_config = load_config(args.general_config)
        
        with open("/biolib/secrets/WANDB_API_KEY", "r") as f:
            wandb_api_key = f.read().strip().strip('"')
        
        if wandb_api_key:
            print("Trying to login to wandb...")
            wandb.login(key=wandb_api_key)
        else:
            logger.error("WANDB_API_KEY not found in environment variables")
            exit(1)

        wandb.init(
            project=general_config['wandb']['project'],
            entity=general_config['wandb']['entity'],
            name=general_config['wandb']['name'],
            mode=general_config['wandb']['mode'],
            config={**model_config, **dataset_config},
            notes=f"Standard dataset",
            group=args.wandb_group
        )
    else:
        logger.info("WANDB logging disabled")

    # Data parameters
    dataset_root = dataset_config['data']['dataset_root']
    
    dataset_filename = dataset_config['data'].get('dataset_filename', None)
    
    dataset_filename_train = dataset_config['data'].get('dataset_filename_train', None)
    dataset_filename_val = dataset_config['data'].get('dataset_filename_val', None)
    
    max_length = dataset_config['data'].get('max_length', 512)
    mask_prob = dataset_config['data'].get('mask_prob', 0.15)
    seed = dataset_config['data'].get('seed', 42)
    
    if dataset_filename is not None:
        data_path = os.path.join(dataset_root, dataset_filename)
        train_split = dataset_config['data'].get('train_split', 0.95)
        
        logger.info(f"Splitting dataset into train and validation sets with train split {train_split}.")
        data_path_train, data_path_val, train_size, val_size = split_sequences(input_fasta=data_path, output_dir=os.path.dirname(data_path), max_length=max_length, train_split=train_split)
        
        logger.info("DONE.")
        logger.info(f"Train path: {data_path_train}, Val path: {data_path_val}")
        logger.info(f"Train size: {train_size}, Val size: {val_size}")
    else:
        assert dataset_filename_train is not None and dataset_filename_val is not None, "dataset_filename or dataset_filename_train and dataset_filename_val must be provided"
        logger.info("Using provided train and validation paths.")
        data_path_train = os.path.join(dataset_root, dataset_filename_train)
        data_path_val = os.path.join(dataset_root, dataset_filename_val)
        
        train_size = dataset_config['data'].get('train_size', None)
        val_size = dataset_config['data'].get('val_size', None)
        
    # Model parameters
    embedding_dim = model_config['model']['embedding_dim']
    hidden_dim = model_config['model']['hidden_dim']
    num_layers = model_config['model']['num_layers']
    kernel_size = model_config['model']['kernel_size']
    dropout_rate = model_config['model']['dropout_rate']
    pad_idx = NUCLEOTIDE_VOCAB['<PAD>']
    vocab_size = len(NUCLEOTIDE_VOCAB)

    # Training parameters
    batch_size = model_config['training']['batch_size']
    num_epochs = model_config['training']['num_epochs']
    learning_rate = model_config['training']['learning_rate']
    model_save_path = model_config['training'].get('model_save_path', './models')

    tokenizer = CharTokenizer(NUCLEOTIDE_VOCAB)
    
    training_set = MaskedRNAIterableDataset(data_path=data_path_train, 
                            tokenizer=tokenizer, 
                            vocab=NUCLEOTIDE_VOCAB, 
                            max_length=max_length, 
                            mask_prob=mask_prob,                      
                            seed=seed,
                            padding=True)
    validation_set = MaskedRNAIterableDataset(data_path=data_path_val, 
                            tokenizer=tokenizer, 
                            vocab=NUCLEOTIDE_VOCAB, 
                            max_length=max_length,
                            mask_prob=mask_prob,
                            seed=seed,
                            padding=True)
   
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
    
    model = MLM1DConv(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, kernel_size=kernel_size, dropout_rate=dropout_rate, pad_idx=pad_idx)
    
    logger.info("Model summary:")
    logger.info(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params}")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train(model, training_loader, validation_loader, optimizer, num_epochs=num_epochs, device=device, model_save_path=model_save_path, train_size=train_size, val_size=val_size, batch_size=batch_size, log_wandb=args.wandb)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)