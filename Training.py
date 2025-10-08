import torch
import torch.nn as nn
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
import psutil
import os
from tqdm import tqdm


def train_with_memory_monitoring(model, dataloader, epochs=3, modelSelected='restnet50'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            encoded, decoded = model(data)
            loss = criterion(decoded, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping untuk stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print SEBELUM delete
            if batch_idx % 10 == 0:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, Memory: {current_memory:.2f} MB')
            
            # Memory management - hapus variabel setelah print
            del encoded, decoded
            if batch_idx % 5 == 0:  # Kurangi frekuensi gc.collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / batch_count
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint setiap epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f'../data/models/checkpoint/{modelSelected}_checkpoint_epoch_{epoch}.pth')
        
        # Clear memory setelah setiap epoch
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model

def simple_train(model, dataloader, epochs=2):
    """Training function yang lebih sederhana dan robust"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            try:
                # Forward pass
                encoded, decoded = model(data)
                loss = criterion(decoded, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print progress setiap 10 batch
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Current Loss: {loss.item():.6f}')
                
                # Cleanup
                del encoded, decoded, loss
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch} finished. Average Loss: {avg_loss:.6f}')
        
        # Memory cleanup di akhir epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model

def train_with_progress(model, dataloader, epochs=2):
    """Training dengan progress bar dan memory management yang baik"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for data, target in pbar:
            try:
                # Forward pass
                encoded, decoded = model(data)
                loss = criterion(decoded, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{(total_loss/batch_count):.6f}'
                })
                
                # Cleanup
                del encoded, decoded
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory error, skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Cleanup di akhir epoch
        del pbar
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model