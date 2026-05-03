import os
import torch
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir, METRICS_TO_CALCULATE):
        self.METRICS_TO_CALCULATE = METRICS_TO_CALCULATE
        self.writer = SummaryWriter(log_dir=log_dir)

    def init_metrics(self):
        return {key: [] for key in self.METRICS_TO_CALCULATE.keys()}

    def calculate_metrics(self, fake, HR, metrics_dict):
        for key, metric in self.METRICS_TO_CALCULATE.items():
            metrics_dict[key].append(metric(HR, fake))
        
        return metrics_dict

    def log_metrics(self, metrics_dict, task, epoch):
        for key in self.METRICS_TO_CALCULATE.keys():
            metric = sum(metrics_dict[key]) / len(metrics_dict[key] or [1])
            self.writer.add_scalar(f"{key}_{task}", metric, epoch)
            print(f'{key}_{task}: {metric}')
        print()

    def log_loss(self, c_loss, g_loss, epoch):
        self.writer.add_scalar("C_loss", c_loss, epoch)
        self.writer.add_scalar("G_loss", g_loss, epoch)
        print(f'C_loss: {c_loss}')
        print(f'G_loss: {g_loss}')
        print()

    @staticmethod
    def log_images(img_path, save_path, epoch, G):
        device = next(G.parameters()).device
        for img_name in os.listdir(img_path):
            img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
            
            # downsample
            new_shape = (img.width // 4, img.height // 4)
            img_low = img.resize(new_shape, Image.BICUBIC)
            
            img_tensor = torch.from_numpy(
                np.array(img_low, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                output = G(img_tensor)
                output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255)
                output = output.byte().cpu().numpy()

            # save generated image
            base_name = img_name.split('_')[0]
            output_path = os.path.join(save_path, f'{base_name}_{epoch:04d}.png')
            Image.fromarray(output).save(output_path)