import argparse

from torch.utils.data import DataLoader, IterableDataset
from epsilon_transformers.training.logger import StructuredLogger
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.training.dataloader import get_dataloader_and_loss_lower_bound_from_process, get_dataloader_from_data
import torch
import numpy as np
import copy
from tqdm import tqdm
import sys

from transformer_lens import HookedTransformer, HookedTransformerConfig
import json
import yaml
import os
from torch.nn import functional as F
from epsilon_transformers.training.generate_data import load_process_data

import wandb


def train_epoch(model, optimizer, dataset, scheduler=None):
    model.train()

    epoch_losses = []

    for input_sequences, target_sequences in dataset:
        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(input_sequences)

        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_sequences.reshape(-1).to(torch.int64)

        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = loss.reshape(batch_size, seq_length)

        # Backpropagation
        loss.mean().backward()

        # Perform optimization step
        optimizer.step()

        # Store the loss for this batch
        epoch_losses.append(loss.detach())

    #if scheduler:
        # Pass the mean loss to the scheduler
        #scheduler.step(torch.mean(torch.cat(epoch_losses)))

    # Compute and return the mean loss per context position across all batches
    return torch.concat(epoch_losses).mean(dim=0)

def train_epoch_all(model, optimizer, dataset, scheduler=None):
    model.train()

    X, Y, probs = dataset.validation_data()
    logits = model(X)
    batch_size, seq_length, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = Y.reshape(-1).to(torch.int64)
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = loss.reshape(batch_size, seq_length)
    loss = loss * probs.unsqueeze(1)
    loss = loss.sum(dim=0) / probs.sum()
    loss.mean().backward()
    optimizer.step()
    #if scheduler:
    #    scheduler.step(loss.mean())
    return loss

def validate_epoch_all(model, dataset, scheduler=None):
    model.eval()

    with torch.no_grad():
        X, Y, probs = dataset.validation_data()
        logits = model(X)
        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = Y.reshape(-1).to(torch.int64)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = loss.reshape(batch_size, seq_length)
        # multiply the loss (batch_size, seq_length) by the probabilities (batch_size) to get the weighted loss (batch_size, seq_length)
        loss = loss * probs.unsqueeze(1) / probs.sum()
        if scheduler:
            scheduler.step(loss.mean())
            #scheduler.step()
        return loss.sum(dim=0)

def validate_epoch_sample(model, dataset):
    pass
    # TODO: implement validate_epoch_sample

def save_model_config(logger, model):
    hooked_model_config_dict = copy.deepcopy(model.cfg.to_dict())
    hooked_model_config_dict['dtype'] = str(hooked_model_config_dict['dtype'])
    with open(os.path.join(logger.base_dir, 'hooked_model_config.json'), 'w') as f:
        json.dump(hooked_model_config_dict, f, indent=4)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(args):
    if args.device == 'cuda':
        if torch.cuda.is_available():
            return torch.device(f'cuda:{args.gpu_id}')
        else:
            print("CUDA is not available. Falling back to CPU.")
            return torch.device('cpu')
    elif args.device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("MPS is not available. Falling back to CPU.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ProductDataset(IterableDataset):
    def __init__(self, dataset_0, dataset_1, dataset_1_vocab_size):
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1
        self.tokens_per_epoch = self.dataset_0.tokens_per_epoch
        self.dataset_1_vocab_size = dataset_1_vocab_size

    def __iter__(self):
        for (X_0, Y_0), (X_1, Y_1) in zip(self.dataset_0, self.dataset_1):
            yield self._product(X_0, Y_0, X_1, Y_1)

    def validation_data(self):
        X_0, Y_0, probs_0 = self.dataset_0.validation_data()
        sample_inds = torch.multinomial(self.dataset_1.probs, X_0.shape[0], replacement=True)
        sample_1 = self.dataset_1.transformer_inputs[sample_inds]
        X_1 = sample_1[:, :-1]
        Y_1 = sample_1[:, 1:]
        probs_1 = self.dataset_1.probs[sample_inds]

        X_prod, Y_prod = self._product(X_0, Y_0, X_1, Y_1)
        return X_prod, Y_prod, probs_0 * probs_1

    def _product(self, X_0, Y_0, X_1, Y_1):
        X_prod = X_0 * self.dataset_1_vocab_size + X_1
        Y_prod = Y_0 * self.dataset_1_vocab_size + Y_1
        return X_prod, Y_prod


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with specific hyperparameters.')
    parser.add_argument('--config', type=str, required=True, help='Path to run configuration file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel CUDA execution')
    parser.add_argument('--gpu_id', type=int, default=0, required=False, help='GPU ID to use for this run')

    args = parser.parse_args()

    config = load_config(args.config)
    logger = StructuredLogger(config['experiment_dir'])
    set_seed(42)

    if config['global_config']['wandb']:
        wandb.init(project=f"{config['global_config']['wandb_project']}_{config['global_config']['sweep_id']}",
                   name=config['run_id'])

    # Set device
    
    if args.parallel:
        device = f'cuda:{args.gpu_id}'
    else:
        device = config['global_config']['device']
    #print(f"Using device: {device}")

    val_every = config['global_config']['val_every']
    train_type = config['train_config'].get('train_type', 'normal')
    save_every = config['global_config'].get('save_every', 1)
    # Parse process parameters

    # Try to load pre-generated data
    # process_data_0 = load_process_data(config, config['global_config']['process_dir'], process_config_override={'name': 'mess3', 'x': config['process_config']['x'], 'a': config['process_config']['a']})
    process_data_0 = load_process_data(config, config['global_config']['process_dir'], process_config_override={'name': 'tom_quantum', 'alpha': config['process_config']['alpha'], 'beta': config['process_config']['beta']})
    process_data_1 = load_process_data(config, config['global_config']['process_dir'], process_config_override={'name': 'identity'})

    print("process_data:", process_data_0, process_data_1)

    if process_data_0 is None or process_data_1 is None:
        raise Exception('preload mess3 and tom_quantum')
    
    # Data was pre-generated, load it
    dataloader_0, d_vocab_0 = get_dataloader_from_data(
        process_data_0['transformer_inputs'],
        process_data_0['probs'],
        config['train_config']['batches_per_epoch'],
        config['train_config']['batch_size'],
        device
    )
    loss_lower_bound_0 = torch.from_numpy(process_data_0['loss_lower_bound']).to(device)
    print('dataloader created')
    sys.stdout.flush()
    dataloader_1, d_vocab_1 = get_dataloader_from_data(
        process_data_1['transformer_inputs'],
        process_data_1['probs'],
        config['train_config']['batches_per_epoch'],
        config['train_config']['batch_size'],
        device
    )
    loss_lower_bound_tom = torch.from_numpy(process_data_1['loss_lower_bound']).to(device)
    print('dataloader created')
    sys.stdout.flush()


    np.savetxt('loss_lower_bound_mess3.txt', loss_lower_bound_0.cpu().numpy(), fmt='%f', delimiter=',', header='loss_lower_bound')
    np.savetxt('loss_lower_bound_tom.txt', loss_lower_bound_tom.cpu().numpy(), fmt='%f', delimiter=',', header='loss_lower_bound')

    config['model_config']['device'] = config['global_config']['device']
    config['model_config']['d_vocab'] = d_vocab_0 * d_vocab_1
    config['model_config']['dtype'] = getattr(torch, config['model_config']['dtype'])

    hooked_model_config = HookedTransformerConfig(**config['model_config'])
    model = HookedTransformer(hooked_model_config)
    #model = torch.compile(model)
    logger.log({"status": "model loaded"})
    save_model_config(logger, model)
    if train_type == 'all':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['train_config']['learning_rate'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train_config']['learning_rate'])
    
    if config['global_config']['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000, cooldown=200, threshold=1e-6,
        )
        # implement cosine annealing warm restarts
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer, T_0=200, T_mult=2, eta_min=1e-7, last_epoch=-1, verbose=True
        #)
    else:
        scheduler = None
    print('MODEL DEVICE:', next(model.parameters()).device, type(next(model.parameters()).device))
    # print the device of the dataloader
    #print('DATALOADER DEVICE:', dataloader.device, type(dataloader.device))
    bar = tqdm(range(config['train_config']['n_epochs']), desc="Training", unit="epoch")

    # save initial model checkpoint
    
    num_tokens_seen = 0
    # do validation before starting the epoch loop

    dataloader = ProductDataset(dataloader_0, dataloader_1, d_vocab_1)
    loss_lower_bound = loss_lower_bound_0 + loss_lower_bound_tom
    print(f'llbs: {loss_lower_bound} {loss_lower_bound_0} {loss_lower_bound_tom}')

    val_loss_per_ctx_pos = validate_epoch_all(model, dataloader)
    print(val_loss_per_ctx_pos.mean().item());sys.stdout.flush()
    val_loss_per_ctx_pos = val_loss_per_ctx_pos / loss_lower_bound
    mean_val_loss = val_loss_per_ctx_pos.mean().item()
    logger.log_epoch(-1, num_tokens_seen, 
                     None, 
                     val_loss_per_ctx_pos.tolist(), 
                     optimizer.param_groups[0]['lr'])
    logger.save_model_checkpoint(model, "0")

    for i in bar:
        if train_type == 'all':
            loss_per_ctx_pos = train_epoch_all(model, optimizer, dataloader, scheduler) / loss_lower_bound
        else:
            loss_per_ctx_pos = train_epoch(model, optimizer, dataloader, scheduler) / loss_lower_bound
        mean_loss = loss_per_ctx_pos.mean().item()
        
        if val_every is not None and i % val_every == 0:
            # TODO: implement logic for val_type
            val_loss_per_ctx_pos = validate_epoch_all(model, dataloader, scheduler) / loss_lower_bound
            mean_val_loss = val_loss_per_ctx_pos.mean().item()
            bar.set_postfix(loss=f"{mean_loss:.4f}", val_loss=f"{mean_val_loss:.4f}")
        else:
            val_loss_per_ctx_pos = None
            bar.set_postfix(loss=f"{mean_loss:.4f}")

        num_tokens_seen += dataloader.tokens_per_epoch
        if save_every is not None and i % save_every == 0:
            logger.save_model_checkpoint(model, f"{num_tokens_seen}")
        logger.log_epoch(i, num_tokens_seen, loss_per_ctx_pos.tolist(), 
                         val_loss_per_ctx_pos.tolist() if val_loss_per_ctx_pos is not None else None, 
                         optimizer.param_groups[0]['lr'])

if __name__ == "__main__":
    main()
