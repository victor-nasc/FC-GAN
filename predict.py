import os
import argparse
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

from FCGAN.data.dataset import SRDataset
from FCGAN.networks.generator import Generator
from FCGAN.configuration.config import load_config


def load_generator(cfg):
    G = Generator().to(cfg['device'])
    
    # load best checkpoint available
    job_path = Path(cfg['save_folder']) / Path(cfg['job'])
    checkpoint_path = job_path / 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_path)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"Please train the model first")
    
    checkpoint = sorted(checkpoint_files)[-1]    
    checkpoint = checkpoint.split('_')[-1].split('.')[0]

    G.load_state_dict(torch.load(checkpoint_path / f'G_{checkpoint}.pth', 
                                 weights_only=True, 
                                 map_location=cfg['device']), strict=True)
    
    print('>>> Evaluating using checkpoint:', checkpoint)
    return G


def main(config_path, ds_path):
    """Predict using the DownGAN model and save generated images"""

    cfg = load_config(config_path)
    G = load_generator(cfg)
    G.eval()

    save_path = Path("./tests") / Path(cfg['job']) / 'predictions'
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path / ds_path.split('/')[-2] / ds_path.split('/')[-1] # TODO use just [-1]
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print(f">>> The directory {save_path} already exists")
        save_path = Path(f"{save_path}_{int(time.time())}")
        os.makedirs(save_path, exist_ok=True)
    print(f">>> Saving predicted images to: {save_path}")

    test_ds = SRDataset(ds_path, train=False, return_name=True)
    test_dl = DataLoader(test_ds, 
                         batch_size=1, 
                         shuffle=False, 
                         num_workers=1)

    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_dl):
            LR = data[0].to(cfg['device'])
            HR = data[1].to(cfg['device'])
            img_name = data[2]

            fake_HR = G(LR)

            for j in range(fake_HR.shape[0]):
                output = fake_HR[j].detach().cpu()
                output = output.permute(1, 2, 0).clamp(0, 1)
                output = (output * 255).byte().numpy()

                name = img_name[j]
                Image.fromarray(output).save(Path(save_path) / name)
        
        print(f"Processed batch {i + 1}/{len(test_dl)}", end='\r')

        print(f"\nDone. Total time: {time.time() - start_time:.2f} seconds")
        print(f"Predicted images are saved in: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict DownGAN")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('-d', '--ds_path', type=str, required=False, help='Path to the dataset')
    args = parser.parse_args()

    main(args.config, args.ds_path)
