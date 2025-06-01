import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
os.environ["WANDB_SILENT"] = "true"
import logging
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from pretrain_mlm_1dcnn import MLM1DConv, NUCLEOTIDE_VOCAB

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('finetune_cnn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define VOCAB_SIZE and PAD_IDX based on NUCLEOTIDE_VOCAB
VOCAB_SIZE = len(NUCLEOTIDE_VOCAB)
PAD_IDX = NUCLEOTIDE_VOCAB['<PAD>']

def tokenize_and_pad_sequence(sequence, vocab, max_seq_len, pad_idx):
    tokens = [vocab.get(nuc, vocab.get('N', 4)) for nuc in sequence.upper()]
    tokens = tokens[:max_seq_len]
    input_ids_list = tokens
    attention_mask_list = [1] * len(tokens)
    padding_needed = max_seq_len - len(tokens)
    input_ids_list += [pad_idx] * padding_needed
    attention_mask_list += [0] * padding_needed
    return torch.tensor(input_ids_list, dtype=torch.long), torch.tensor(attention_mask_list, dtype=torch.long)

# Task-specific CNN heads for different RNA analysis tasks
class SecondaryStructureCNNHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: [batch_size, input_dim, seq_len]
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Back to [batch_size, seq_len, 64]
        x = x.transpose(1, 2)
        
        # Project to output classes at each position
        x = self.fc(x)
        
        return x

class SpliceSiteCNNHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, mask=None):
        # x: [batch_size, input_dim, seq_len]
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Apply mask if provided
        if mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, seq_len]
            mask = mask.unsqueeze(1).float()
            x = x * mask  # Zero out padding positions
        
        # Global max pooling
        x = self.pool(x).squeeze(-1)  # [batch_size, 128]
        
        # Classification layer
        x = self.fc(x)
        
        return x

class NcRNAFamilyCNNHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(512, 256)  # 512 = 256*2 (concat avg and max pool)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x, mask=None):
        # x: [batch_size, input_dim, seq_len]
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            x = x * mask
        
        # Global pooling (combine avg and max for better features)
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [batch_size, 256]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [batch_size, 256]
        
        # Concatenate the pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch_size, 512]
        
        # Final classification layers
        x = self.fc1(pooled)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class ModificationSiteCNNHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, mask=None):
        # x: [batch_size, input_dim, seq_len]
        
        # Apply deeper CNN stack for capturing modification patterns
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            x = x * mask
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # [batch_size, 64]
        
        # Output projection
        x = self.fc(x)
        
        return x

class FineTuningCNN(nn.Module):
    def __init__(self, pretrained_model_path, embedding_dim, hidden_dim, num_layers, kernel_size, dropout_rate, 
                 vocab_size, pad_idx, num_classes_task, task_type="sequence", task_name=None, random_init=False):
        super().__init__()
        # Use MLM1DConv as the backbone
        self.cnn_backbone = MLM1DConv(vocab_size, embedding_dim, hidden_dim, num_layers, kernel_size, dropout_rate, pad_idx)
        if not random_init:
            try:
                pretrained_state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
                # Adjust keys if necessary, e.g., if the pretrained model has a different prefix for layers
                # For MLM1DConv, the output layer is self.output, not self.fc
                backbone_load_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('output.')}
                
                # Handle potential missing keys or mismatched shapes, especially for the embedding layer if vocab size changed
                model_dict = self.cnn_backbone.state_dict()
                # Filter out unnecessary keys from pretrained_state_dict
                pretrained_dict_filtered = {k: v for k, v in backbone_load_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict_filtered)
                self.cnn_backbone.load_state_dict(model_dict, strict=False) # Use strict=False to allow partial loading
                
                if len(pretrained_dict_filtered) < len(backbone_load_state_dict):
                    print("Warning: Some layers from the pretrained model were not loaded. This might be due to architecture mismatches or naming differences.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Proceeding with random initialization for the backbone.")
                self.cnn_backbone = MLM1DConv(vocab_size, embedding_dim, hidden_dim, num_layers, kernel_size, dropout_rate, pad_idx) # Re-initialize if loading failed
        else:
            print("Random initialization selected for backbone. No pretrained weights loaded.")

        self.task_type = task_type
        self.task_name = task_name
        self.hidden_dim = hidden_dim # This should be the output dim of the backbone's conv layers
        self.num_classes_task = num_classes_task
        
        # Initialize the appropriate task head based on task_name
        if task_name == 'secondary_structure':
            self.task_head = SecondaryStructureCNNHead(hidden_dim, num_classes_task)
        elif task_name == 'splice_site':
            self.task_head = SpliceSiteCNNHead(hidden_dim, num_classes_task)
        elif task_name == 'ncrna_family':
            self.task_head = NcRNAFamilyCNNHead(hidden_dim, num_classes_task)
        elif task_name == 'modification_site':
            self.task_head = ModificationSiteCNNHead(hidden_dim, num_classes_task)
        else:
            # Default task head (simple linear projection after pooling)
            self.task_head = nn.Linear(hidden_dim, num_classes_task)
        
        self.random_init = random_init

        if not random_init and 'pretrained_dict_filtered' in locals() and len(pretrained_dict_filtered) > 0:
            for param in self.cnn_backbone.embedding.parameters():
                param.requires_grad = False
            for param in self.cnn_backbone.conv_layers.parameters(): # Changed from self.cnn_backbone.cnn
                param.requires_grad = False
            print("Backbone frozen (embedding and conv_layers). Only task_head is trainable initially.")
        elif random_init:
            print("Backbone is trainable from the start (random init mode). Only task_head is new.")
        else:
            print("Backbone is trainable as pretrained weights failed to load or were not fully applied.")

    def forward(self, src_seq, attention_mask=None):
        # Get embeddings
        embedded = self.cnn_backbone.embedding(src_seq)  # [batch, seq_len, embed_dim]
        
        # Reshape for 1D convolution
        conv_input = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # Apply convolutions from MLM1DConv backbone
        # The conv_layers in MLM1DConv is a Sequential module
        conv_output = self.cnn_backbone.conv_layers(conv_input)  # [batch, hidden_dim, seq_len]
        
        if self.task_type == "sequence":
            # For sequence-level tasks, use task-specific CNN head
            # All sequence-level task heads handle their own pooling
            logits = self.task_head(conv_output, attention_mask)
            return logits
        
        elif self.task_type == "token":
            # For token-level tasks, use token-level task head
            logits = self.task_head(conv_output)
            return logits
        
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
            
    def unfreeze_backbone(self, unfreeze_embedding=True, unfreeze_cnn=True):
        if unfreeze_embedding:
            for param in self.cnn_backbone.embedding.parameters():
                param.requires_grad = True
            print("Unfroze embedding layer.")
        if unfreeze_cnn:
            for param in self.cnn_backbone.conv_layers.parameters(): # Changed from self.cnn_backbone.cnn
                param.requires_grad = True
            print("Unfroze CNN layers (conv_layers).")

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_params(model):
    """Print the model architecture and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    backbone_params = sum(p.numel() for p in model.cnn_backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model.cnn_backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.task_head.parameters())
    head_trainable = sum(p.numel() for p in model.task_head.parameters() if p.requires_grad)
    
    print("\n=== MODEL ARCHITECTURE ===")
    print(f"Task type: {model.task_type}, Task name: {model.task_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print("\n--- Backbone ---")
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Trainable backbone parameters: {backbone_trainable:,} ({backbone_trainable/backbone_params:.2%})")
    print("\n--- Task Head ---")
    print(f"Task head parameters: {head_params:,}")
    print(f"Trainable task head parameters: {head_trainable:,} ({head_trainable/head_params:.2%})")
    
    # Print task head architecture
    print("\nTask Head Architecture:")
    task_head_str = str(model.task_head)
    # Format the output for better readability
    for line in task_head_str.split('\n'):
        print(f"  {line}")
    print("========================\n")

# Dataset classes (identical to CNN version)
class SpliceSiteFineTuneDataset(Dataset):
    def __init__(self, tsv_path, vocab, max_seq_len, pad_idx):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.unique_labels = sorted(self.data['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.unique_labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = str(item['sequence'])
        input_ids, attention_mask = tokenize_and_pad_sequence(sequence, self.vocab, self.max_seq_len, self.pad_idx)
        label_val = item['label']
        label_idx = self.label_to_idx[label_val]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label_idx, dtype=torch.long)}

class NcRNAFamilyFineTuneDataset(Dataset):
    def __init__(self, tsv_path, vocab, max_seq_len, pad_idx):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.unique_labels = sorted(self.data['label'].unique())
        self.label_to_idx = {int(label): idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: int(label) for idx, label in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = str(item['sequence'])
        input_ids, attention_mask = tokenize_and_pad_sequence(sequence, self.vocab, self.max_seq_len, self.pad_idx)
        family_name = item['label']
        label_idx = self.label_to_idx[family_name]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(label_idx, dtype=torch.long)}

class SecondaryStructureFineTuneDataset(Dataset):
    LABEL_PAD_IDX = -100
    def __init__(self, csv_path, vocab, max_seq_len, pad_idx):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.struct_to_idx = {'(': 0, '.': 1, ')': 2}
        self.num_classes = len(self.struct_to_idx)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = str(item['sequence'])
        structure_str = str(item.get('secondary_structure', ''))
        input_ids, attention_mask = tokenize_and_pad_sequence(sequence, self.vocab, self.max_seq_len, self.pad_idx)
        labels = []
        effective_seq_len = attention_mask.sum().item()
        for i in range(self.max_seq_len):
            if i < len(structure_str) and i < effective_seq_len:
                labels.append(self.struct_to_idx.get(structure_str[i], self.struct_to_idx['.']))
            else:
                labels.append(self.LABEL_PAD_IDX)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': torch.tensor(labels, dtype=torch.long)}

class ModificationSiteFineTuneDataset(Dataset):
    def __init__(self, tsv_path, vocab, max_seq_len, pad_idx):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.label_columns = [col for col in self.data.columns if col.startswith('labels_')]
        self.num_classes = len(self.label_columns)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = str(item['sequences'])
        input_ids, attention_mask = tokenize_and_pad_sequence(sequence, self.vocab, self.max_seq_len, self.pad_idx)
        labels = torch.tensor([item[col] for col in self.label_columns], dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def custom_collate_fn_finetune(batch):
    # Filter out items with empty (all-pad) input sequences
    filtered_batch = [item for item in batch if item['attention_mask'].sum().item() > 0]
    if len(filtered_batch) == 0:
        raise ValueError("All sequences in batch are empty after filtering. Check your dataset for empty or invalid sequences.")
    keys = filtered_batch[0].keys()
    collated_batch = {}
    for key in keys:
        if torch.is_tensor(filtered_batch[0][key]):
            collated_batch[key] = torch.stack([item[key] for item in filtered_batch])
        else:
            collated_batch[key] = [item[key] for item in filtered_batch]
    return collated_batch

def evaluate_model_finetune(model, dataloader, criterion, device, task_type, num_classes_task):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    if dataloader is None or len(dataloader) == 0:
        return float('inf'), 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            if task_type == "sequence":
                loss = criterion(outputs, labels)
                total_loss += loss.item() * input_ids.size(0)

                if isinstance(criterion, nn.BCEWithLogitsLoss): # Multi-label case (e.g., modification_site)
                    preds_binary = (torch.sigmoid(outputs) > 0.5).float()
                    correct_per_sample = (preds_binary == labels).all(dim=1) # [batch_size] boolean
                    total_correct += correct_per_sample.sum().item() # sum of true values
                else: # Multi-class single-label case (e.g., splice_site, ncrna_family)
                    _, predicted_class_indices = torch.max(outputs, 1) # [batch_size]
                    total_correct += (predicted_class_indices == labels).sum().item()
                
                total_samples += labels.size(0) # Number of sequences in batch
            elif task_type == "token":
                loss = criterion(outputs.reshape(-1, num_classes_task), labels.reshape(-1))
                total_loss += loss.item() * input_ids.size(0)
                active_loss = labels.view(-1) != SecondaryStructureFineTuneDataset.LABEL_PAD_IDX
                active_logits = outputs.view(-1, num_classes_task)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                if active_logits.size(0) > 0:
                    _, predicted_tokens = torch.max(active_logits, 1)
                    total_correct += (predicted_tokens == active_labels).sum().item()
                    total_tokens += active_labels.size(0)
                total_samples += input_ids.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    if task_type == "sequence":
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    elif task_type == "token":
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    else:
        accuracy = 0.0
    return avg_loss, accuracy

def train_model_finetune(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, device, model_save_path, task_type, num_classes_task, unfreeze_epoch, initial_lr, early_stopping_patience=5, min_delta=1e-4, log_wandb=False):
    logger.info(f"Starting fine-tuning for {num_epochs} epochs on device: {device}")
    best_avg_val_loss = float('inf')
    backbone_unfrozen = False
    epochs_without_improvement = 0
    
    # For tracking metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0
        total_correct = 0
        total_samples = 0
        
        if not backbone_unfrozen and unfreeze_epoch != -1 and (epoch + 1) >= unfreeze_epoch and not model.random_init:
            logger.info(f"Epoch {epoch + 1}: Unfreezing CNN backbone.")
            model.unfreeze_backbone(unfreeze_embedding=True, unfreeze_cnn=True)
            optimizer = optim.Adam(model.parameters(), lr=initial_lr / 10 if initial_lr > 1e-5 else 1e-5)
            logger.info("Optimizer re-initialized for all trainable parameters.")
            logger.info("\n=== PARAMETER COUNTS AFTER UNFREEZING ===")
            trainable_params = count_parameters(model)
            total_params = sum(p.numel() for p in model.parameters())
            backbone_trainable = sum(p.numel() for p in model.cnn_backbone.parameters() if p.requires_grad)
            logger.info(f"Total trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
            logger.info(f"Backbone trainable parameters: {backbone_trainable:,}")
            logger.info("===========================================\n")
            backbone_unfrozen = True
        
        batch_iterator = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            unit="batch", 
            leave=True
        )
        
        for batch_idx, batch in enumerate(batch_iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            
            if task_type == "sequence":
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    preds_binary = (torch.sigmoid(outputs) > 0.5).float()
                    correct_per_sample = (preds_binary == labels).all(dim=1)
                    total_correct += correct_per_sample.sum().item()
                else:
                    _, predicted_class_indices = torch.max(outputs, 1)
                    total_correct += (predicted_class_indices == labels).sum().item()
                    
                total_samples += labels.size(0)
                
            elif task_type == "token":
                loss = criterion(outputs.reshape(-1, num_classes_task), labels.reshape(-1))
                
                # Calculate accuracy for token-level tasks
                active_loss = labels.view(-1) != SecondaryStructureFineTuneDataset.LABEL_PAD_IDX
                active_logits = outputs.view(-1, num_classes_task)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                if active_logits.size(0) > 0:
                    _, predicted_tokens = torch.max(active_logits, 1)
                    total_correct += (predicted_tokens == active_labels).sum().item()
                    total_samples += active_labels.size(0)
            else:
                raise ValueError("Invalid task type for loss calculation")
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            # Update progress bar with current loss
            if batch_idx % 100 == 0:
                batch_iterator.set_postfix({'Train Loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Store metrics
        train_losses.append(avg_epoch_train_loss)
        train_accs.append(train_accuracy)
        
        logger.info(f"Epoch {epoch+1} train loss: {avg_epoch_train_loss:.4f}, train accuracy: {train_accuracy:.4f}")
        
        # Handle case where validation dataloader is None or empty
        if val_dataloader is not None and len(val_dataloader) > 0:
            avg_epoch_val_loss, val_accuracy = evaluate_model_finetune(model, val_dataloader, criterion, device, task_type, num_classes_task)
            val_losses.append(avg_epoch_val_loss)
            val_accs.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1} val loss: {avg_epoch_val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
            
            # Log to wandb if enabled
            if log_wandb:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': avg_epoch_train_loss,
                        'train/accuracy': train_accuracy,
                        'val/loss': avg_epoch_val_loss,
                        'val/accuracy': val_accuracy,
                        'epoch': epoch
                    })
                except Exception as e:
                    logger.error(f"Error logging to wandb: {e}")
            
            # Early stopping logic with validation loss
            if avg_epoch_val_loss < best_avg_val_loss - min_delta:
                best_avg_val_loss = avg_epoch_val_loss
                epochs_without_improvement = 0
                logger.info(f"Validation loss improved to {best_avg_val_loss:.4f}. Saving best model.")
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        else:
            # If no validation data, use training loss for early stopping
            logger.info(f"No validation data available. Using training loss for model selection.")
            
            # Log to wandb if enabled
            if log_wandb:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': avg_epoch_train_loss,
                        'train/accuracy': train_accuracy,
                        'epoch': epoch
                    })
                except Exception as e:
                    logger.error(f"Error logging to wandb: {e}")
            
            if avg_epoch_train_loss < best_avg_val_loss - min_delta:
                best_avg_val_loss = avg_epoch_train_loss
                epochs_without_improvement = 0
                logger.info(f"Training loss improved to {best_avg_val_loss:.4f}. Saving best model.")
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in training loss for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    break
    
    # Plot and save training curves if we have validation data
    if len(val_losses) > 0:
        try:
            # Save directory should be the directory containing the model
            save_dir = os.path.dirname(model_save_path)
            if save_dir == '':
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            
            plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir)
            logger.info(f"Training curves saved to {save_dir}")
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")
        
    # Write epoch metrics to CSV
    save_dir = os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.'
    csv_path = os.path.join(save_dir, 'finetune_training_metrics.csv')
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        for i in range(len(train_losses)):
            val_loss = val_losses[i] if i < len(val_losses) else ''
            val_acc = val_accs[i] if i < len(val_accs) else ''
            writer.writerow([i+1, train_losses[i], train_accs[i], val_loss, val_acc])
    logger.info(f"Fine-tuning metrics CSV saved to {csv_path}")
    
    final_loss_type = "validation" if val_dataloader and len(val_dataloader) > 0 else "training"
    logger.info(f"Fine-tuning completed. Best model saved to {model_save_path} with best {final_loss_type} loss: {best_avg_val_loss:.4f}")

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_dir):
    """Plot training and validation curves and save to file."""
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained CNN on downstream RNA tasks.")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to the pretrained_cnn.pt model.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training set (TSV/CSV).")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation set (TSV/CSV).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test set (TSV/CSV).")
    parser.add_argument("--model_save_path", type=str, default="output/finetuned_cnn.pt", help="Path to save the fine-tuned model.")
    parser.add_argument("--task", type=str, required=True, choices=['splice_site', 'ncrna_family', 'secondary_structure', 'modification_site'], help="Downstream task type.")
    parser.add_argument("--task_name", type=str, default="finetune", help="Name of the task for wandb logging.")
    parser.add_argument("--wandb_group", type=str, default="finetune", help="Wandb group name.")
    
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of nucleotide embeddings (must match pretrained).")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of CNN hidden states (must match pretrained).")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of CNN layers (must match pretrained).")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for CNN layers (must match pretrained).")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for CNN and embedding (must match pretrained).")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate for fine-tuning the head.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for fine-tuning.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of fine-tuning epochs.")
    parser.add_argument("--unfreeze_epoch", type=int, default=0, help="Epoch after which to unfreeze backbone (e.g., 3 means unfreeze at start of epoch 3). Set to 0 for immediate unfreezing of backbone, -1 to keep frozen.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--random_init", action="store_true", help="If set, do not load pretrained weights and use random initialization.")
    parser.add_argument("--evaluate_only_on_test", action="store_true", help="If set, skip training and only evaluate on the test set using the model from --model_save_path.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum change in validation loss to qualify as an improvement.")
    parser.add_argument("--wandb", "-w", action="store_true", help="Enable wandb logging")
    parser.add_argument("--general_config", "-g", type=str, default="configs/general.yml", help="Path to general YAML config file (for wandb settings)")

    args,_ = parser.parse_known_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    
    # Set up wandb logging if enabled
    log_wandb = False
    if args.wandb:
        try:
            import wandb
            # Try to load config and initialize wandb
            try:
                from yaml import safe_load
                with open(args.general_config, 'r') as f:
                    general_config = safe_load(f)
                
                # Try to load API key from file or environment variable
                try:
                    if os.path.exists("/biolib/secrets/WANDB_API_KEY"):
                        with open("/biolib/secrets/WANDB_API_KEY", "r") as f:
                            wandb_api_key = f.read().strip().strip('"')
                            wandb.login(key=wandb_api_key)
                    else:
                        # Try to use environment variable if exists
                        wandb.login()
                
                    # Initialize wandb
                    run_name = f"finetune_{args.task_name}_{os.path.basename(args.test_path).split('.')[0]}"
                    wandb.init(
                        project=general_config['wandb'].get('project', 'rna-foundation-model'),
                        entity=general_config['wandb'].get('entity', 'your-entity'),
                        name=run_name,
                        group=args.wandb_group,
                        mode=general_config['wandb'].get('mode', 'online'),
                        config={
                            "task": args.task,
                            "task_name": args.task_name,
                            "model_type": "cnn",
                            "pretrained": not args.random_init,
                            "embedding_dim": args.embedding_dim,
                            "hidden_dim": args.hidden_dim,
                            "num_layers": args.num_layers,
                            "kernel_size": args.kernel_size,
                            "dropout_rate": args.dropout_rate,
                            "learning_rate": args.learning_rate,
                            "batch_size": args.batch_size,
                            "max_seq_len": args.max_seq_len,
                            "num_epochs": args.num_epochs,
                            "unfreeze_epoch": args.unfreeze_epoch,
                            "train_dataset": Path(args.train_path).name,
                            "val_dataset": Path(args.val_path).name,
                            "test_dataset": Path(args.test_path).name
                        },
                        notes=f"Fine-tuning CNN on {args.task_name} task"
                    )
                    log_wandb = True
                except Exception as e:
                    logger.error(f"Error logging in to wandb: {e}")
                    logger.error("Continuing without wandb logging")
            except Exception as e:
                logger.error(f"Error loading general config: {e}")
                logger.error("Continuing without wandb logging")
        except ImportError:
            logger.error("wandb package not installed. Continuing without wandb logging.")
    else:
        logger.info("WANDB logging disabled")
        
    dataset_class = None
    if args.task == 'splice_site':
        dataset_class = SpliceSiteFineTuneDataset
    elif args.task == 'ncrna_family':
        dataset_class = NcRNAFamilyFineTuneDataset
    elif args.task == 'secondary_structure':
        dataset_class = SecondaryStructureFineTuneDataset
    elif args.task == 'modification_site':
        dataset_class = ModificationSiteFineTuneDataset
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Determine num_classes_task from train_dataset.
    # This is needed for model initialization even in evaluate_only_on_test mode.
    # main.py will ensure a valid train_path is always passed.
    try:
        # Ensure train_path is valid for deriving num_classes
        if not Path(args.train_path).exists():
            print(f"Error: Train path {args.train_path} for num_classes derivation not found.")
            return
        _train_dataset_for_num_classes = dataset_class(args.train_path, NUCLEOTIDE_VOCAB, args.max_seq_len, PAD_IDX)
        num_classes_task = _train_dataset_for_num_classes.num_classes
        del _train_dataset_for_num_classes
    except FileNotFoundError:
        print(f"Error: Train path {args.train_path} not found during num_classes derivation. Cannot determine num_classes.")
        return
    except Exception as e:
        print(f"Error loading train dataset for num_classes: {e}")
        return

    if args.task == 'modification_site':
        task_type_for_model = "sequence"
    elif args.task in ['splice_site', 'ncrna_family']:
        task_type_for_model = "sequence"
    elif args.task == 'secondary_structure':
        task_type_for_model = "token"
    else:
        raise ValueError(f"Unknown task: {args.task}")

    model = FineTuningCNN(
        pretrained_model_path=args.pretrained_model_path, # Backbone path
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout_rate,
        vocab_size=VOCAB_SIZE,
        pad_idx=PAD_IDX,
        num_classes_task=num_classes_task, # Derived from train_dataset
        task_type=task_type_for_model,
        task_name=args.task,  # Pass the task name to select the appropriate CNN head
        random_init=args.random_init
    ).to(device)

    # Print model architecture and parameter counts
    print_model_params(model)

    # Setup criterion (needed for both training and evaluation)
    if args.task == 'modification_site':
        criterion = nn.BCEWithLogitsLoss().to(device)
    elif args.task in ['splice_site', 'ncrna_family']:
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.task == 'secondary_structure':
        criterion = nn.CrossEntropyLoss(ignore_index=SecondaryStructureFineTuneDataset.LABEL_PAD_IDX).to(device)
    else:
        print(f"Error: Unknown task {args.task} for criterion setup.")
        return

    test_loss, test_acc = float('nan'), float('nan') # Initialize for the case where only training happens and test fails

    if args.evaluate_only_on_test:
        if not Path(args.model_save_path).exists():
            print(f"Error: --evaluate_only_on_test specified, but model file {args.model_save_path} not found.")
            return
        
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        
        try:
            if not Path(args.test_path).exists():
                print(f"Error: Test path {args.test_path} not found for evaluation-only mode.")
                return
            test_dataset = dataset_class(args.test_path, NUCLEOTIDE_VOCAB, args.max_seq_len, PAD_IDX)
        except FileNotFoundError:
            print(f"Error: Test path {args.test_path} not found.") # Should be caught by Path.exists()
            return
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return

        if len(test_dataset) == 0:
            print(f"Error: Test dataset at {args.test_path} is empty.")
            return
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn_finetune)

    else: # Full training and then evaluation mode
        try:
            # Ensure all paths are valid before creating datasets
            if not Path(args.train_path).exists():
                print(f"Error: Train path {args.train_path} not found.")
                return
            if not Path(args.val_path).exists():
                print(f"Error: Validation path {args.val_path} not found.")
                return
            if not Path(args.test_path).exists(): # This is the first test file in a potential series
                print(f"Error: Initial test path {args.test_path} not found.")
                return

            train_dataset = dataset_class(args.train_path, NUCLEOTIDE_VOCAB, args.max_seq_len, PAD_IDX)
            val_dataset = dataset_class(args.val_path, NUCLEOTIDE_VOCAB, args.max_seq_len, PAD_IDX)
            test_dataset = dataset_class(args.test_path, NUCLEOTIDE_VOCAB, args.max_seq_len, PAD_IDX)
        except FileNotFoundError as e: # Should be caught by checks above, but as a fallback
            print(f"Error: Dataset file not found: {e.filename}")
            return
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return

        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            print("Error: One or more datasets (train, val, or initial test) are empty. Please check paths and formats.")
            if len(test_dataset) == 0 : print(f"Specifically, initial test_dataset at {args.test_path} is empty.")
            return

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn_finetune, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn_finetune)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn_finetune)
        print(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

        # Optimizer setup
        if args.unfreeze_epoch == 0 and not model.random_init:
            print("Unfreezing backbone immediately.")
            model.unfreeze_backbone(unfreeze_embedding=True, unfreeze_cnn=True)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
        
        # Criterion already set up above

        train_model_finetune(
            model, 
            train_dataloader, 
            val_dataloader, 
            optimizer, 
            criterion, 
            args.num_epochs, 
            device, 
            args.model_save_path, 
            task_type_for_model, 
            num_classes_task, 
            args.unfreeze_epoch, 
            args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            log_wandb=log_wandb
        )
        
        if not Path(args.model_save_path).exists():
            print(f"Error: Trained model {args.model_save_path} not found after training. Cannot evaluate.")
            # Metrics will remain NaN, and will be written to CSV as such.
        else:
            model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    
    # Common evaluation logic (test_dataloader and criterion must be defined and model loaded)
    # Ensure test_dataloader is available; it's set in both branches or an error would have occurred.
    if 'test_dataloader' in locals() and test_dataloader is not None:
        logger.info(f"\nEvaluating model on test set: {args.test_path}...")
        test_loss, test_acc = evaluate_model_finetune(model, test_dataloader, criterion, device, task_type_for_model, num_classes_task)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        if log_wandb:
            wandb.log({
                "test/loss": test_loss,
                "test/acc": test_acc
            })
    else:
        logger.info(f"Skipping final evaluation on {args.test_path} as test_dataloader was not set up (e.g. file not found or empty).")
        # test_loss, test_acc remain NaN

    all_labels = []
    all_preds = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Final Test Evaluation"): # Added tqdm here
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask)
            if args.task == 'modification_site':
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            elif args.task in ['splice_site', 'ncrna_family']:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
            elif args.task == 'secondary_structure':
                probs = torch.softmax(outputs, dim=2).cpu().numpy().reshape(-1, num_classes_task)
                preds = np.argmax(probs, axis=1)
                labels = batch['labels'].cpu().numpy().reshape(-1)
                mask = labels != SecondaryStructureFineTuneDataset.LABEL_PAD_IDX
                labels = labels[mask]
                preds = preds[mask]
                probs = probs[mask]
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs)
    if args.task == 'modification_site':
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        if all_labels.ndim == 1:
            all_labels = all_labels.reshape(-1, num_classes_task)
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(-1, num_classes_task)
        all_labels = all_labels.astype(int)
        all_preds = all_preds.astype(int)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        # Per-label AUROC
        mod_labels = ['Am', 'Cm', 'Gm', 'Tm', 'm1A', 'm5C', 'm5U', 'm6A', 'm6Am', 'm7G', 'Φ', 'I']
        per_label_auroc = []
        for i in range(num_classes_task):
            try:
                auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            except Exception:
                auroc = float('nan')
            per_label_auroc.append(auroc)
        avg_auroc = np.nanmean(per_label_auroc)
    elif args.task in ['splice_site', 'ncrna_family']:
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = all_labels.astype(int).flatten()
        all_preds = all_preds.astype(int).flatten()
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    elif args.task == 'secondary_structure':
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = all_labels.astype(int).flatten()
        all_preds = all_preds.astype(int).flatten()
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        f1_macro = f1_micro = precision_macro = recall_macro = float('nan')
    results_csv = "finetune_results.csv"
    metrics = {
        'task': args.task,
        'train_path': args.train_path,
        'val_path': args.val_path,
        'test_path': args.test_path,
        'model_save_path': args.model_save_path,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'kernel_size': args.kernel_size,
        'dropout_rate': args.dropout_rate,
        'max_seq_len': args.max_seq_len,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'unfreeze_epoch': args.unfreeze_epoch,
        'learning_rate': args.learning_rate,
        'random_init': args.random_init,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }
    # Add per-label AUROC and average AUROC to metrics if modification_site
    if args.task == 'modification_site':
        for i, auroc in enumerate(per_label_auroc):
            label_name = mod_labels[i] if i < len(mod_labels) else f'label_{i}'
            metrics[f'auroc_label_{i}'] = auroc
            metrics[f'auroc_labelname_{label_name}'] = auroc
        metrics['auroc_avg'] = avg_auroc
        
        # Log modification site specific metrics to wandb
        if log_wandb:
            wandb.log({
                'test/f1_macro': f1_macro,
                'test/f1_micro': f1_micro,
                'test/precision_macro': precision_macro,
                'test/recall_macro': recall_macro,
                'test/avg_auroc': avg_auroc
            })
            # Log per-label metrics
            for i, label_name in enumerate(mod_labels[:num_classes_task]):
                wandb.log({
                    f'test/auroc_{label_name}': per_label_auroc[i]
                })
                
    elif args.task in ['splice_site', 'ncrna_family']:
        # Log classification metrics to wandb
        if log_wandb:
            wandb.log({
                'test/f1_macro': f1_macro,
                'test/f1_micro': f1_micro,
                'test/precision_macro': precision_macro,
                'test/recall_macro': recall_macro
            })
            
    elif args.task == 'secondary_structure':
        # Log secondary structure metrics to wandb
        if log_wandb:
            wandb.log({
                'test/f1_macro': f1_macro,
                'test/f1_micro': f1_micro,
                'test/precision_macro': precision_macro,
                'test/recall_macro': recall_macro
            })
            
    write_header = not os.path.exists(results_csv)
    with open(results_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    logger.info(f"Results written to {results_csv}")
    
    # Finish wandb run
    if log_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()