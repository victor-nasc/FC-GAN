import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.autograd import Variable

from FCGAN.metrics.metrics import SSIM, PSNR
from FCGAN.metrics.perceptual_loss import PerceptualLoss
from FCGAN.utils.logger import Logger

torch.autograd.set_detect_anomaly(True)

METRICS_TO_CALCULATE = {
    "MAE": nn.L1Loss(),
    "SSIM": SSIM,
    "PSNR": PSNR, 
}

class FrequencyConditionedGAN:
    """Implements Frequency Conditioned GAN with WGAN-GP training"""

    def __init__(self, cfg, G, C, G_optimizer, C_optimizer):
        self.num_steps = 0
        self.cfg = cfg
        self.G = G
        self.C = C
        self.G_optimizer = G_optimizer
        self.C_optimizer = C_optimizer
        self.job_path = Path(cfg["save_folder"]) / Path(cfg["job"])
        self.VGG_loss = PerceptualLoss(cfg["device"], cfg["vgg_layers"])
        self.logger = Logger(self.job_path / 'logs', METRICS_TO_CALCULATE)

    def save_model(self, epoch):
        save_path = self.job_path / 'checkpoints'
        torch.save(self.G.state_dict(), save_path / f'G_{epoch:04d}.pth')
        torch.save(self.G_optimizer.state_dict(), save_path / f'G_optim_{epoch:04d}.pth')
        
        if self.cfg['2nd_stage']:
            torch.save(self.C.state_dict(), save_path / f'C_{epoch:04d}.pth')
            torch.save(self.C_optimizer.state_dict(), save_path / f'C_optim_{epoch:04d}.pth')

        print(f'>>> Model saved at {epoch:04d}.pth')

    def _critic_train_iteration(self, fake, HR):
        self.C_optimizer.zero_grad()
        
        if self.cfg['freq_sep']:
            _, fake = self._separate_frequencies(fake)
            _, HR = self._separate_frequencies(HR)

        GP = self._gp(HR, fake)
        c_loss = torch.mean(self.C(fake)) - torch.mean(self.C(HR)) + self.cfg["gp_lambda"] * GP

        c_loss.backward(retain_graph=True)
        self.C_optimizer.step()
        
        return c_loss.item() 

    def _generator_train_iteration(self, fake, HR):
        self.G_optimizer.zero_grad()

        if self.cfg['freq_sep']:
            fake_l, fake_h = self._separate_frequencies(fake)
            HR_l, _ = self._separate_frequencies(HR)

            g_loss = self.cfg["content_lambda"] * nn.L1Loss()(fake_l, HR_l)

            if self.cfg['2nd_stage']:
                g_loss -= self.cfg["adv_lambda"] * torch.mean(self.C(fake_h))
                g_loss += self.cfg["percep_lambda"] * self.VGG_loss(fake, HR)
        else:
            g_loss = self.cfg["content_lambda"] * nn.L1Loss()(fake, HR)
            
            if self.cfg['2nd_stage']:
                g_loss -= self.cfg["adv_lambda"] * torch.mean(self.C(fake))
                g_loss += self.cfg["percep_lambda"] * self.VGG_loss(fake, HR) 

        g_loss.backward()
        self.G_optimizer.step()

        return g_loss.item()

    def _gp(self, real, fake):
        current_batch_size = real.size(0)

        # calculate interpolation
        alpha = torch.rand(current_batch_size, 1, 1, 1, device=self.cfg["device"])
        alpha = alpha.expand_as(real)

        interpolated = alpha * real.data + (1 - alpha) * fake.data
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        critic_interpolated = self.C(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=self.cfg["device"]),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(self.cfg["batch_size"], -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def _separate_frequencies(self, image):
        padding = self.cfg['filter_size'] // 2
        low_filter = nn.AvgPool2d(self.cfg['filter_size'], stride=1, padding=0)
        padding_layer = nn.ReplicationPad2d(padding)

        padded_image = padding_layer(image)
        image_low = low_filter(padded_image)

        image_high = image - image_low
        return image_low, image_high
    
    def _train_epoch(self, train_dl, val_dl, epoch):
        print(80*"=")
        
        train_metrics = self.logger.init_metrics()
        c_loss, g_loss = 0.0, 0.0

        for i, data in enumerate(train_dl):
            start_time = time.time()
            LR = data[0].to(self.cfg["device"])
            HR = data[1].to(self.cfg["device"])

            fake = self.G(LR)   
            
            if self.cfg['2nd_stage']:
                c_loss += self._critic_train_iteration(fake, HR)

            g_loss += self._generator_train_iteration(fake, HR)
                
            train_metrics = self.logger.calculate_metrics(fake, HR, train_metrics)
            self.num_steps += 1

            # print(f'train step: {i}/{len(train_dl)} Time: {time.time() - start_time:.2f}s')
            # break
        
        with torch.no_grad():
            val_metrics = self.logger.init_metrics()

            for i, data in enumerate(val_dl):
                start_time = time.time()
                LR = data[0].to(self.cfg["device"])
                HR = data[1].to(self.cfg["device"])

                fake = self.G(LR).detach() 

                val_metrics = self.logger.calculate_metrics(fake, HR, val_metrics)

                # print(f'val step: {i}/{len(val_dl)} Time: {time.time() - start_time:.2f}s')
                # break

            # log metrics and loss
            c_loss = c_loss / len(train_dl)
            g_loss = g_loss / len(train_dl)

            self.logger.log_loss(c_loss, g_loss, epoch) 
            self.logger.log_metrics(train_metrics, "train", epoch)
            self.logger.log_metrics(val_metrics, "val", epoch)

        if epoch % self.cfg["save_every"] == 0 and epoch != 0:
            self.logger.log_images(self.cfg["sample_images"], self.job_path / 'images', epoch, self.G)
            self.save_model(epoch)

        print(f'epoch: {epoch}/{self.cfg["epochs"]}')

    def train(self, train_dl, val_dl, epoch_start=0):
        for epoch in range(epoch_start, self.cfg["epochs"]):
            start_time = time.time()
            self._train_epoch(train_dl, val_dl, epoch)
            print(f"Time: {time.time() - start_time:.2f}s")

