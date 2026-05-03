import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from FCGAN.data.dataset import SRDataset
from FCGAN.networks.critic import Critic

from FCGAN.networks.generator import Generator
from FCGAN.fc_gan import FrequencyConditionedGAN
from FCGAN.configuration.config import load_config


def train(config_path):
    cfg = load_config(config_path)
        
    job_path = Path(cfg['save_folder']) / Path(cfg['job'])
    os.makedirs(job_path, exist_ok=True)
    os.makedirs(job_path / 'checkpoints', exist_ok=True)
    os.makedirs(job_path / 'logs', exist_ok=True)
    os.makedirs(job_path / 'images', exist_ok=True)

    # data loading
    train_ds = SRDataset(Path(cfg['data_path']) / 'train', train=True)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)

    val_ds = SRDataset(Path(cfg['data_path']) / 'val', train=False)
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size']//4, shuffle=False, num_workers=4)

    # model initialization
    C = Critic().to(cfg['device'])
    G = Generator().to(cfg['device'])
    print(f'>>> Number of parameters in critic:    {sum(p.numel() for p in C.parameters())}')
    print(f'>>> Number of parameters in generator: {sum(p.numel() for p in G.parameters())}')

    G_optimizer = torch.optim.Adam(G.parameters(), cfg['g_lr'], betas=(0.9, 0.99))
    C_optimizer = torch.optim.Adam(C.parameters(), cfg['c_lr'], betas=(0.9, 0.99))

    # load latest checkpoint if exists
    checkpoint_path = job_path / 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_path)
    if checkpoint_files:
        # checkpoints are in format G_0001.pth, G_optim_0001.pth, C_0001.pth, C_optim_0001.pth ...
        checkpoint = sorted(checkpoint_files)[-1] 
        checkpoint = checkpoint.split('_')[-1].split('.')[0]

        G.load_state_dict(torch.load(checkpoint_path / f'G_{checkpoint}.pth', weights_only=True, map_location=cfg['device']))
        G_optimizer.load_state_dict(torch.load(checkpoint_path / f'G_optim_{checkpoint}.pth', weights_only=True, map_location=cfg['device']))
        if cfg['2nd_stage']:
            try:
                C.load_state_dict(torch.load(checkpoint_path / f'C_{checkpoint}.pth', weights_only=True, map_location=cfg['device']))
                C_optimizer.load_state_dict(torch.load(checkpoint_path / f'C_optim_{checkpoint}.pth', weights_only=True, map_location=cfg['device']))
            except:
                print('>>> Critic checkpoint not found, skipping critic loading')
        print(f'>>> Loaded checkpoint {checkpoint}')
        checkpoint = int(checkpoint) + 1
    else:
        checkpoint = 0
        print('>>> Training from scratch')

    # training
    trainer = FrequencyConditionedGAN(cfg, G, C, G_optimizer, C_optimizer)
    trainer.train(train_dl, val_dl, epoch_start=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    
    train(args.config)