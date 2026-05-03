import torch
import numpy as np
import lpips
from pytorch_msssim import ssim
from torchmetrics.image.fid import FrechetInceptionDistance as fid
# from pysteps.utils.spectral import rapsd

def SSIM(real: torch.Tensor, fake: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    fake = torch.clamp(fake, 0, data_range)

    return ssim(real, fake, data_range=data_range).detach().cpu().item()


def PSNR(real: torch.Tensor, fake: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    fake = torch.clamp(fake, 0, data_range)

    mse = torch.mean((real - fake) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * torch.log10(data_range ** 2 / mse).detach().cpu().item()


def LPIPS(real: torch.Tensor, fake: torch.Tensor, lpips_model: lpips, data_range: float = 1.0) -> torch.Tensor:
    fake = torch.clamp(fake, 0, data_range)

    return torch.mean(lpips_model(real, fake), dim=0).detach().cpu().item()
    

# def RALSD(real: torch.Tensor, fake: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
#     """Radially Averaged Log-Spectrum Distance (RALSD)"""
#     fake = torch.clamp(fake, 0, data_range)
#     real_np = real.detach().cpu().numpy()
#     fake_np = fake.detach().cpu().numpy()

#     total_ralsd = 0.0
#     B, C = real_np.shape[:2]
    
#     for b in range(B): 
#         img_ralsd = 0.0
        
#         for c in range(C):
#             P_real = rapsd(real_np[b, c], fft_method=np.fft)
#             P_fake = rapsd(fake_np[b, c], fft_method=np.fft)
            
#             log_diff = 10 * np.log10((P_real + 1e-8) / (P_fake + 1e-8)) # avoid log(0)
#             img_ralsd += np.sqrt(np.mean(np.square(log_diff)))
        
#         total_ralsd += (img_ralsd / C)
    
#     return total_ralsd / B


def update_FID(real: torch.Tensor, fake: torch.Tensor, fid_model: fid, data_range: float = 1.0) -> None:
    # FID should be updated for each batch
    # the metric result should be computed after all batches are processed!
    fake = torch.clamp(fake, 0, data_range)

    with torch.no_grad():
        fid_model.update(real, real=True)
        fid_model.update(fake, real=False)


def FID(fid_model: fid) -> float:
    # FID should be computed after all batches are processed!
    return fid_model.compute().item()