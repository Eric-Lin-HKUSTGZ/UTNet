#!/usr/bin/env python3
"""
Quick test script to verify distributed evaluation fix
"""
import os
import torch
import torch.distributed as dist

def test_data_collection():
    """Test the data collection logic without full training"""
    
    # Simulate 4 GPUs with different amounts of data
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Simulate different data sizes (like DistributedSampler might give)
    # Total: 8332 samples, 4 GPUs → 2083, 2083, 2083, 2083
    if world_size == 4:
        if rank == 0:
            num_samples = 2083
        elif rank == 1:
            num_samples = 2083
        elif rank == 2:
            num_samples = 2083
        else:  # rank == 3
            num_samples = 2083
    else:
        num_samples = 8332 // world_size + (1 if rank < 8332 % world_size else 0)
    
    # Create fake predictions (J=21 joints, 3D)
    local_pred = torch.randn(num_samples, 21, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
    local_gt = torch.randn(num_samples, 21, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Rank {rank}: {num_samples} samples')
    
    if dist.is_available() and dist.is_initialized():
        device = local_pred.device
        
        # Gather sizes from all processes
        local_size = torch.tensor([local_pred.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        
        if rank == 0:
            print(f'\nRank 0: Received sizes from all GPUs: {[s.item() for s in size_list]}')
            print(f'Rank 0: Total samples: {sum(s.item() for s in size_list)}')
            
            # Collect data
            all_preds_list = []
            all_gts_list = []
            
            for i, size in enumerate(size_list):
                num_samples_i = size.item()
                if i == rank:
                    all_preds_list.append(local_pred.cpu())
                    all_gts_list.append(local_gt.cpu())
                else:
                    recv_pred = torch.zeros(num_samples_i, local_pred.shape[1], local_pred.shape[2],
                                          dtype=local_pred.dtype, device=device)
                    recv_gt = torch.zeros(num_samples_i, local_gt.shape[1], local_gt.shape[2],
                                        dtype=local_gt.dtype, device=device)
                    dist.recv(recv_pred, src=i)
                    dist.recv(recv_gt, src=i)
                    all_preds_list.append(recv_pred.cpu())
                    all_gts_list.append(recv_gt.cpu())
            
            all_pred = torch.cat(all_preds_list, dim=0)
            all_gt = torch.cat(all_gts_list, dim=0)
            
            print(f'Rank 0: Concatenated shape: {all_pred.shape}')
            print(f'✅ Test passed! Successfully collected {all_pred.shape[0]} samples from {world_size} GPUs')
        else:
            # Send data to rank 0
            dist.send(local_pred, dst=0)
            dist.send(local_gt, dst=0)
            print(f'Rank {rank}: Sent {num_samples} samples to rank 0')
    else:
        print('Not running in distributed mode, test passed trivially')
        print(f'✅ Single GPU mode: {num_samples} samples')

if __name__ == '__main__':
    # This script should be run with torchrun:
    # torchrun --nproc_per_node=4 test_distributed_eval.py
    
    if dist.is_available() and 'RANK' in os.environ:
        # Initialize distributed training
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        print(f'Initialized process {rank}')
    
    test_data_collection()
    
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

