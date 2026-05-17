import torch
from .scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def noisy_image(n=1, img_shape=(28, 28), device=None):
    """Generates a batch of n pure-noise images. Output shape: (n, 1, H, W)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.randn((n, 1, *img_shape), device=device)


def generate_image(unet, scheduler: NoiseScheduler, stochasticity=1.0, n_images=1,
                   return_intermediates=False, use_time=True):
    """
    Runs the full reverse diffusion chain and returns the denoised image tensor.
    """
    intermediates = []
    unet.eval()

    model_device = next(unet.parameters()).device

    x = noisy_image(n_images, device=model_device)
    if return_intermediates:
        intermediates.append(x.detach().cpu())

    with torch.no_grad():
        for t in range(scheduler.T - 1, 0, -1):
            z = noisy_image(n_images, device=model_device) if t > 1 else torch.zeros_like(x)

            alpha     = scheduler.alpha(t).to(model_device)
            alpha_bar = scheduler.alpha_bar(t).to(model_device)
            sigma_t   = stochasticity * torch.sqrt(scheduler.beta(t).to(model_device))

            if use_time:
                t_batch = torch.full((n_images,), t, device=model_device, dtype=torch.long)
                noise_pred = unet(x, t_batch)
            else:
                noise_pred = unet(x, None)

            # IMPORTANT: this update should happen exactly once.
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
            ) + sigma_t * z

            if return_intermediates:
                intermediates.append(x.detach().cpu())

    if return_intermediates:
        return x, intermediates
    return x