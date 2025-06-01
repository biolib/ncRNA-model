import torch.nn as nn
import torch

import os
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
        logging.FileHandler('mlm_transformer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', '-m',type=str, default='configs/model/transformer.yml', help='Path to model YAML config file')
parser.add_argument('--dataset_config', '-d', type=str, default='configs/data/basic_rnacentral_10k.yml', help='Path to dataset YAML config file')
parser.add_argument('--general_config', '-g', type=str, default='configs/transformer.yml', help='Path to general YAML config file')
parser.add_argument('--wandb', '-w', action='store_true', help='Enable wandb logging')
parser.add_argument('--wandb_group', type=str, default='group_name', help='Wandb group name.')

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer from the original transformer paper.
    """
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer so it's not updated during training
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embedding_dim]
        return self.pe[:x.size(1), :].transpose(0, 1)

class MLMTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dim_feedforward, num_layers, num_heads, 
                 dropout_rate, pad_idx, max_seq_len=5000):
        """
        Transformer encoder for masked language modeling.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings (should be divisible by num_heads)
            dim_feedforward: Hidden dimension in feed-forward layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            pad_idx: Padding token index
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(MLMTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True  # Important: makes input shape [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection layer
        self.output = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Scale embeddings by sqrt(embedding_dim) as per original transformer paper
        self.embedding_scale = math.sqrt(embedding_dim)
        
    def encode(self, x, attention_mask=None):
        # Token embedding
        embedded = self.embedding(x) * self.embedding_scale  # [batch, seq_len, embedding_dim]
        pos_encoded = self.pos_encoding(embedded)
        
        # Add positional encoding
        embedded = embedded + pos_encoded
        embedded = self.dropout(embedded)
        
        # Convert attention mask to src_key_padding_mask
        src_key_padding_mask = (attention_mask == 1) if attention_mask is not None else None
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(
            embedded, 
            src_key_padding_mask=src_key_padding_mask
        )
        return encoder_output
        
    def decode(self, encoder_output):
        # Apply the final layer
        return self.output(encoder_output)
        
    def forward(self, x, attention_mask=None):
        encoder_output = self.encode(x, attention_mask)
        return self.decode(encoder_output)
    
    def get_attention_weights(self, x, attention_mask=None):
        """Get attention weights from all layers (for analysis)."""
        # This would require modifying the transformer to return attention weights
        # For now, just a placeholder
        return None
    

def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, device, model_save_path, train_size, val_size, batch_size, log_wandb, early_stopping_patience=30, min_delta=1e-4):
    # Ignore -100 index in loss calculation
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    logger.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} of {num_epochs}")
        # Training
        model.train()
        
        train_loss = 0
        
        correct_preds = 0
        total_masked = 0
        
        for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, total=math.ceil(train_size/batch_size) if train_size is not None else None, unit="batch")):
            input_ids, labels, att_mask = data
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            att_mask = att_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, att_mask)
            
            # TODO: cross entropy labels should be one-hot encoded? No. DONE
            
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
                running_vloss += vloss
                
                vmasked_positions = (vlabels.view(-1) != -100)
                vpredictions = voutputs.view(-1, voutputs.size(-1)).argmax(dim=-1)
                vcorrect = (vpredictions[vmasked_positions] == vlabels.view(-1)[vmasked_positions]).sum().item()
                vcorrect_preds += vcorrect
                vtotal_masked += vmasked_positions.sum().item()

        avg_val_acc = vcorrect_preds / vtotal_masked
        avg_val_loss = running_vloss / (i + 1)
        
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
    torch.save(model.state_dict(), f'{model_save_path}/1dcnn_model_acc_{avg_train_acc:.4f}_vacc_{avg_val_acc:.4f}_loss_{avg_train_loss:.4f}_vloss_{avg_val_loss:.4f}.pth')
    
        
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
    
    # Dataset splitting
    # If the dataset isn't already in two files for each of train and val, split it
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
    dim_feedforward = model_config['model']['dim_feedforward']
    num_layers = model_config['model']['num_layers']
    num_heads = model_config['model']['num_heads']
    dropout_rate = model_config['model']['dropout_rate']
    
    pad_idx = NUCLEOTIDE_VOCAB['<PAD>']
    vocab_size = len(NUCLEOTIDE_VOCAB)

    # Training parameters
    batch_size = model_config['training']['batch_size']
    num_epochs = model_config['training']['num_epochs']
    learning_rate = model_config['training']['learning_rate']
    model_save_path = model_config['training'].get('model_save_path', './models')
    
    os.makedirs(model_save_path, exist_ok=True)

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
    
    model = MLMTransformer(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        pad_idx=pad_idx
    )
    
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