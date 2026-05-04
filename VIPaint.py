"""VIPaint sampler for Stable Diffusion 3."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sd3_impls
from sd3_impls import kl_divergence
from sd3_infer import CFGDenoiser, CFG_SCALE, SkipLayerCFGDenoiser
from lpips.loss.lpips_masked import LPIPS


def tensor_to_np(tensor, scale=True):
    """Convert a (B, C, H, W) tensor to a uint8 NumPy array of shape (B, H, W, C)."""
    tensor = tensor.permute(0, 2, 3, 1).clamp(0, 1)
    return (tensor.detach().cpu().numpy() * (255 if scale else 1)).astype(np.uint8)


def np_to_tensor(np_img, scale=True):
    """Convert a NumPy image to a (B, C, H, W) float32 tensor on the active device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = torch.from_numpy(np_img).float() / (255.0 if scale else 1.0)
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor.to(device=device, dtype=torch.float32)


def append_dims(x, target_dims):
    """Append singleton dims to ``x`` until it has ``target_dims`` dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def original_sample_mix(mask, masked, sample):
    """Composite the generated sample with the original observation outside the mask."""
    sample_img = np.squeeze(tensor_to_np(sample))
    mask_np = np.squeeze(tensor_to_np(mask))
    masked_np = np.squeeze(tensor_to_np(masked))

    if mask_np.max() > 1:
        mask_np = mask_np / 255.0
    if sample_img.dtype != np.uint8:
        sample_img = (sample_img * 255).clip(0, 255).astype(np.uint8)
    if masked_np.dtype != np.uint8:
        masked_np = (masked_np * 255).clip(0, 255).astype(np.uint8)

    mask3 = np.repeat(mask_np[..., None], 3, axis=2)
    mixed = sample_img * (1 - mask3) + masked_np * mask3
    composited_img = mixed.astype(np.uint8)
    return composited_img


class VIPaintSampler(nn.Module):
    """Variational inpainting sampler with DPS refinement on top of SD3."""

    def __init__(self, masked, mask, inferencer, cfg, skip_layer_config=None):
        nn.Module.__init__(self)
        if skip_layer_config is None:
            skip_layer_config = {}
        with torch.no_grad():
            self.mask = np_to_tensor(mask, scale=False)
            self.masked = ((np_to_tensor(masked, scale=True) * 2 - 1.) * self.mask) * 0.5 + 0.5
            self.cfg_scale = inferencer.cfg_scale if hasattr(inferencer, 'cfg_scale') else CFG_SCALE

            self.cfg = cfg
            prompt = cfg.get("prompt", "")
            bounds = cfg.get("bounds", [400, 550])
            self.batch_size = cfg.get("batch_size", 1)
            self.lr = cfg.get("learning_rate", 1e-2)
            self.steps = cfg.get("steps", 50)
            self.l1_weight = cfg.get("l1_weight", 1.0)
            self.lpips_weight = cfg.get("lpips_weight", 0.0)
            self.kl_weight = cfg.get("kl_weight", 1.)
            self.mid_weight = cfg.get("mid_weight", 1.)
            self.dps_scale = cfg.get("dps_scale", 200.0)
            self.rec_loss_weight = cfg.get("rec_weight", 1.0)
            self.K = cfg.get("K", 2)
            self.N = cfg.get("N", 10)
            self.num_dps_runs = cfg.get("num_dps_runs", 2)

            self.hk = 1000 - bounds[0]
            self.h_steps = [0] + np.linspace(1000 - bounds[1], 1000 - bounds[0], self.K).astype(int).tolist() + [1000]

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.all_sigmas = inferencer.get_sigmas(inferencer.sd3.model.model_sampling, 1000).to(self.device)
            self.dps_sigmas = self.all_sigmas[1000 - bounds[0]:1000]
            self.sigmas = self.all_sigmas[self.h_steps]

            print("Preparing sampling loop...")
            print("h_steps:", self.h_steps)
            print("sigmas:", self.sigmas)

            controlnet_cond = None
            latent = inferencer.encode_first_stage(self.masked, vae_cpu=False)
            neg_cond = inferencer.get_cond("")
            print("Getting latent for prompt:", prompt)
            conditioning = inferencer.get_cond(prompt)
            conditioning = inferencer.fix_cond(conditioning)
            neg_cond = inferencer.fix_cond(neg_cond)
            self.extra_args = {
                "cond": conditioning,
                "uncond": neg_cond,
                "cond_scale": self.cfg_scale,
                "controlnet_cond": controlnet_cond,
            }

            denoiser = (
                SkipLayerCFGDenoiser
                if skip_layer_config.get("scale", 0) > 0
                else CFGDenoiser
            )

            timesteps = list(range(0, 1000))
            self.denoiser = denoiser(inferencer.sd3.model, timesteps, skip_layer_config)

            self.latent = latent.half().to(device=self.device)
            self.mask_lt = F.interpolate(self.mask, size=latent.shape[-2:], mode="nearest")
            self.model = inferencer.sd3.model.to(device=self.device)

        self.tau = nn.ParameterList([nn.Parameter(torch.ones_like(latent, dtype=torch.float32)) for _ in range(2 + self.K + self.N)])
        self.gamma = nn.ParameterList([nn.Parameter(torch.ones_like(latent, dtype=torch.float32)) for _ in range(2 + self.K + self.N)])
        self.mu = nn.ParameterList([nn.Parameter(latent + 0.5 * torch.randn_like(latent)) for _ in range(2 + self.K + self.N)])
        self.KL_loss = torch.zeros(1, device=self.device)
        self.rec_loss = torch.zeros(1, device=self.device)
        self.kl_midterm = 0.

        self.dps_lst = list(range(1000 - bounds[0], 1000))
        self.inferencer = inferencer
        self.skip_layer_config = skip_layer_config

        for p in self.model.parameters():
            p.requires_grad = False

        self.lpips_model = LPIPS().eval()
        self.lpips_model = self.lpips_model.to(device=self.device)
        for p in self.lpips_model.parameters():
            p.requires_grad = False

    def reverse_step(self, sigmas, x, i_t, i_s, denoised):
        """Single reverse step: returns (mu, sigma) of q and (prior_mu, prior_sigma) of p."""
        sigma_t = sigmas[i_t]
        sigma_s = sigmas[i_s]

        alpha_t = (1. - sigma_t)
        alpha_s = (1. - sigma_s)
        alpha_ts = (alpha_t / alpha_s)
        v_t = sigma_t ** 2
        v_s = sigma_s ** 2
        v_ts = v_t - (alpha_ts ** 2) * v_s
        v = v_ts * v_s / v_t

        # conditioned from the prior
        prior_sigma = torch.sqrt(v)
        prior_mu = denoised * append_dims(alpha_s * v_ts / v_t, denoised.ndim) + x * append_dims(alpha_ts * v_s / v_t, x.ndim)

        s_step = self.N + i_s
        vi_denoised = denoised * append_dims(torch.sigmoid(self.gamma[s_step]), denoised.ndim) + self.mu[s_step]
        mu = vi_denoised * append_dims(alpha_s * v_ts / v_t, vi_denoised.ndim) + x * append_dims(alpha_ts * v_s / v_t, x.ndim)
        sigma = torch.sigmoid(self.tau[s_step]) * append_dims(prior_sigma, x.ndim)
        return mu, sigma, prior_mu, prior_sigma

    def posterior_q_zt_given_zt1(self, sigmas, z_t1, i, denoised):
        """Closed-form posterior q(z_t | z_{t+1}) used for the mid-term KL."""
        i_t = i
        i_s = i + 1

        sigma_t = sigmas[i_t]
        sigma_s = sigmas[i_s]

        alpha_t = (1. - sigma_t)
        alpha_s = (1. - sigma_s)
        alpha_ts = (alpha_t / alpha_s)

        v_t = sigma_t ** 2
        v_s = sigma_s ** 2
        v_ts = v_t - (alpha_ts ** 2) * v_s
        v = v_ts * v_s / v_t
        prior_sigma = torch.sqrt(v)
        prior_mu = denoised * append_dims(alpha_s * v_ts / v_t, denoised.ndim) + z_t1 * append_dims(alpha_ts * v_s / v_t, z_t1.ndim)

        hk = -self.K - 1
        sigma_hk = sigmas[hk]
        alpha_hk = (1. - sigma_hk)
        v_hk = sigma_hk ** 2
        alpha_shk = (alpha_s / alpha_hk)
        v_shk = v_s - (alpha_shk ** 2) * v_hk
        v_tt = v_shk + (alpha_shk ** 2) * v_hk

        v_post = 1.0 / ((alpha_ts ** 2) / v_ts + (1.0 / v_tt))

        # conditioned from the prior
        sigma_post = torch.sqrt(v_post)
        mu_post = v_post * ((alpha_ts / (v_t - (alpha_ts ** 2) * v_s)) * z_t1 + alpha_shk * self.mu[hk] / ((alpha_shk ** 2) * v_hk + v_shk))

        return mu_post, sigma_post, prior_mu, prior_sigma

    @torch.autocast("cuda", dtype=torch.float16)
    def sample_ancestral(self):
        """Ancestral sampling along the sparse trajectory; updates KL and reconstruction losses."""
        # at time T, all noise
        noise = torch.randn((self.batch_size,) + self.latent.shape[1:], device=self.latent.device, dtype=self.latent.dtype)

        # KL loss at time T KL(q(z_T) || p(z_T))
        alpha_T = 1 - self.sigmas[0]
        alpha_hk = 1 - self.sigmas[1]
        alpha_T_hk = alpha_T / alpha_hk
        mu_T = alpha_T_hk * self.mu[1]

        var_t = self.sigmas[0] ** 2
        var_hk = self.sigmas[1] ** 2
        v_t_hk = var_t - (alpha_T_hk ** 2) * var_hk
        var_T = v_t_hk + (alpha_T_hk ** 2) * var_hk
        sigma_T = torch.sqrt(var_T)

        x = mu_T + torch.sigmoid(sigma_T) * noise
        size_match = x.new_ones([x.shape[0]])

        # KL loss at time T KL(q(z_T) || p(z_T))
        self.KL_loss = torch.mean(
            kl_divergence(mu_T * size_match, torch.sigmoid(sigma_T) * size_match, torch.zeros_like(x), torch.ones_like(x))
        )

        for i in tqdm(range(len(self.sigmas) - 1)):
            denoised = self.denoiser(x, self.sigmas[i] * size_match, **self.extra_args)
            mu, sigma, prior_mu, prior_sigma = self.reverse_step(self.sigmas, x, i, i + 1, denoised)

            if i < len(self.sigmas) - 2:
                self.KL_loss = self.KL_loss + torch.mean(
                    kl_divergence(mu, sigma, prior_mu, prior_sigma)
                )
                x = mu + sigma * torch.randn_like(x)
            else:
                self.rec_loss = self.recon_loss(denoised, self.sigmas[i])
                x = prior_mu

        return x

    def compute_midterm_kl(self):
        """Stochastic mid-term KL between q and p at N random timesteps."""
        t_random = random.sample(range(2, self.h_steps[1]), self.N)
        t_aug = t_random + [t - 1 for t in t_random]
        steps_subset_full = sorted(self.h_steps + t_aug)

        sigmas = self.all_sigmas[steps_subset_full]
        kl_values = []

        for i in range(1, self.N):
            sig_t1 = 2 * i + 1
            hk = self.N + 1

            sigma_t1 = sigmas[sig_t1]
            alpha_t = 1 - sigma_t1
            alpha_hk = 1 - sigmas[hk]
            alpha_t_hk = alpha_t / alpha_hk
            mu_t1 = alpha_t_hk * self.mu[hk]

            var_t = sigma_t1 ** 2
            var_hk = sigmas[hk] ** 2
            v_t_hk = var_t - (alpha_t_hk ** 2) * var_hk
            var_t1 = v_t_hk + (alpha_t_hk ** 2) * var_hk
            sigma_t1 = torch.sqrt(var_t1)

            noise = torch.randn_like(self.latent)
            z_t1 = mu_t1 + torch.sigmoid(sigma_t1) * noise

            # Denoise to predict z0 from z_{t+1}
            s_in = z_t1.new_ones([z_t1.shape[0]])
            denoised = self.denoiser(z_t1, sigma_t1 * s_in, **self.extra_args)

            # Reverse step: get q(z_t | z_{t+1}) and p(z_t | z_{t+1})
            mu, sigma, prior_mu, prior_sigma = self.posterior_q_zt_given_zt1(sigmas, z_t1, sig_t1, denoised)

            # KL divergence between q and p
            kl = (1000 - self.h_steps[1]) * kl_divergence(mu, sigma, prior_mu, prior_sigma).mean()
            kl_values.append(kl)

        # Return average over N samples
        return torch.stack(kl_values).mean()

    def recon_loss(self, denoised, sigma):
        """Time-weighted reconstruction loss restricted to the mask in latent space."""
        t = 1.0 - sigma
        lam_prime = 2 * (t ** 2 - t)
        w_t = ((t / 2.0) * lam_prime) ** 2

        eps_obs = (sigma * self.latent * self.mask_lt) / t
        eps_hat = (sigma * denoised * self.mask_lt) / t

        loss = (eps_hat - eps_obs) ** 2
        return (w_t * loss).mean()

    def losses(self):
        """Run one forward pass; returns (L2 data term, decoded sample)."""
        samples_z = self.sample_ancestral()
        self.kl_midterm = self.compute_midterm_kl()

        samples = self.inferencer.decode_first_stage(samples_z)
        l2 = 50 * nn.functional.mse_loss(samples * self.mask, self.masked * self.mask).mean()
        return l2, samples

    def optimize(self, num=0, base_dir=None, directory=None, callback=None):
        """Optimize variational parameters and log intermediate diagnostics."""
        opt = torch.optim.Adam(list(self.tau) + list(self.gamma) + list(self.mu), lr=self.lr)
        opt.zero_grad(set_to_none=True)

        steps_list, total_loss_list = [], []
        l1_list, kl_list, mid_kl_list, rec_list = [], [], [], []
        tau_1_list, tau_2_list = [], []

        for step in range(self.steps):
            l1, samples = self.losses()

            kl = self.KL_loss
            mid_kl = self.kl_midterm
            rec = self.rec_loss

            l1 = 100 * self.l1_weight * l1
            kl = 100 * self.kl_weight * kl
            mid_kl = 100 * self.mid_weight * mid_kl
            rec = 100 * self.rec_loss_weight * rec

            print(f" kl: {kl.item()}, mid_kl: {mid_kl.item()}",
                  f"rec: {rec.item()}, step: {step}/{self.steps}")

            loss = l1 + kl + mid_kl + rec
            loss.backward()

            opt.step()
            opt.zero_grad(set_to_none=True)

            steps_list.append(step)
            total_loss_list.append(loss.item())
            l1_list.append(l1.item())
            kl_list.append(kl.item())
            mid_kl_list.append(mid_kl.item())
            rec_list.append(rec.item())
            tau_1_list.append(self.tau[1].mean().item())
            tau_2_list.append(self.tau[2].mean().item())

            with torch.no_grad():
                mu_latent = torch.cat(list(self.mu), dim=0)
                decoded_list = []
                for chunk in mu_latent.split(2):
                    decoded_list.append(
                        self.inferencer.decode_first_stage(chunk)
                    )
                decoded = torch.cat(decoded_list, dim=0)
                mu_np = [tensor_to_np(mui.unsqueeze(0)).squeeze(0) for mui in decoded]

                if (step) % 5 == 0:
                    if callback is not None:
                        callback(
                            progress=step / self.steps,
                            total_loss=loss.item(),
                            l1=l1.item(),
                            kl=kl.item(),
                            recon_loss=rec.item(),
                            sample=tensor_to_np(samples[:1]),
                            mu=mu_np,
                            masked=tensor_to_np(self.masked),
                            mask=tensor_to_np(self.mask, scale=False),
                            dps_sample=None,
                            mid_kl=mid_kl.item()
                        )
                    else:
                        sample_img = np.squeeze(tensor_to_np(samples[:1]))
                        if directory is not None:
                            Image.fromarray(sample_img).save(os.path.join(directory, f"sample_{step:04d}.png"))
                            # ---- loss graph ----
                            plt.figure(figsize=(6, 4))
                            plt.plot(steps_list, total_loss_list, label="Total Loss")
                            plt.plot(steps_list, l1_list, label=f"L1 {self.l1_weight}")
                            plt.plot(steps_list, kl_list, label=f"KL {self.kl_weight}")
                            plt.plot(steps_list, mid_kl_list, label=f"Mid KL {self.mid_weight}")
                            plt.plot(steps_list, rec_list, label=f"Rec {self.rec_loss_weight}")
                            plt.xlabel("Step")
                            plt.ylabel("Loss")
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(os.path.join(directory, f"loss_curve.png"))
                            plt.close()

                            # ---- tau graph ----
                            plt.figure(figsize=(6, 4))
                            plt.plot(steps_list, tau_1_list, marker="o", label=f"tau1 {self.tau[1].mean().item():.4f}")
                            plt.plot(steps_list, tau_2_list, marker="o", label=f"tau2 {self.tau[2].mean().item():.4f}")

                            for x, y in zip(steps_list, tau_1_list):
                                plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=1)

                            for x, y in zip(steps_list, tau_2_list):
                                plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=1)

                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(os.path.join(directory, f"tau_curve.png"))
                            plt.close()
                if step == self.steps - 1:
                    if callback is None:
                        sample_img = np.squeeze(tensor_to_np(samples[:1]))
                        composited_img = original_sample_mix(self.mask, self.masked, samples[:1])
                        if directory is not None:
                            Image.fromarray(sample_img).save(os.path.join(directory, f"{num}_sample_{step:04d}.png"))
                            Image.fromarray(composited_img).save(os.path.join(directory, f"sample_observed.png"))
        return None

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.float16)
    def get_z_ts(self, denoiser):
        """Run an ancestral pass to obtain the starting latent for DPS."""
        y_obs = self.latent
        size_match = y_obs.new_ones([y_obs.shape[0]])
        noise = torch.randn_like(self.mu[0])

        alpha_t = 1 - self.sigmas[0]
        alpha_hk = 1 - self.sigmas[1]
        alpha_t_hk = alpha_t / alpha_hk
        mu_T = alpha_t_hk * self.mu[1]

        var_t = self.sigmas[0] ** 2
        var_hk = self.sigmas[1] ** 2
        v_t_hk = var_t - (alpha_t_hk ** 2) * var_hk
        var_T = v_t_hk + (alpha_t_hk ** 2) * var_hk
        sigma_T = torch.sqrt(var_T)

        z = mu_T + torch.sigmoid(sigma_T) * noise

        for i in tqdm(range(len(self.sigmas) - 2)):
            denoised = denoiser(z, self.sigmas[i] * size_match, **self.extra_args)
            mu, sigma, _, _ = self.reverse_step(self.sigmas, z, i, i + 1, denoised)

            if i < len(self.sigmas) - 2:
                z = mu + sigma * torch.randn_like(z)

        return z

    @torch.autocast("cuda", dtype=torch.float16)
    def dps_sampling(self, num, denoiser, directory: str | None = None):
        """DPS refinement: reverse diffusion with a data-consistency gradient."""
        y_obs_latent = self.latent.detach()
        batch_flag = y_obs_latent.new_ones([y_obs_latent.shape[0]])

        z = self.get_z_ts(denoiser)

        # ====== MAIN LOOP ======
        for i in range(0, len(self.dps_sigmas) - 1):
            sigma_t = self.dps_sigmas[i]
            sigma_s = self.dps_sigmas[i + 1]
            t_tensor = batch_flag * sigma_t

            z = z.detach().requires_grad_()
            z0_hat = denoiser(z, t_tensor, **self.extra_args)

            recon = self.inferencer.decode_first_stage(z0_hat)
            l1 = 20.0 * torch.nn.functional.l1_loss(
                recon * self.mask, self.masked * self.mask)
            lp = self.lpips_model(recon * self.mask, self.masked * self.mask, self.mask).mean()
            recon_loss = l1 + lp

            grad = torch.autograd.grad(
                outputs=recon_loss, inputs=z,
                retain_graph=False, create_graph=False)[0]

            z_corr = (z - self.dps_scale * grad).detach()

            with torch.no_grad():
                z0_hat_corr = denoiser(z_corr, t_tensor, **self.extra_args)
                z_prev = sd3_impls.reverse_sample(z_corr, sigma_t, sigma_s, z0_hat_corr)
            z = z_prev

            rel = (self.dps_scale * grad).abs().mean() / z.abs().mean()
            print(f"step {self.dps_scale} [{i}] σ={sigma_t:.3f} "
                  f"|grad|={grad.abs().mean():.2e} relΔ={rel:.2%}")

            if directory is not None and i % 50 == 0:
                recon_img = np.squeeze(tensor_to_np(recon))
                Image.fromarray(recon_img).save(os.path.join(directory, f"dps_recon_{num}_{i:03d}.png"))

        # ====== FINAL DENOISING ======
        sigma_min = self.dps_sigmas[-1]
        z_hat0 = denoiser(z, batch_flag * sigma_min, **self.extra_args)
        x_out = self.inferencer.decode_first_stage(z_hat0)

        return x_out, z_hat0

    def sample(self, num, base_dir=None, directory=None, callback=None):
        """Run DPS sampling ``self.num_dps_runs`` times and return the final composited image."""
        for i in range(self.num_dps_runs):
            dps_samples, _ = self.dps_sampling(i, self.denoiser, directory=directory)

            sample_img = np.squeeze(tensor_to_np(dps_samples[:1]))
            composited_img = original_sample_mix(self.mask, self.masked, dps_samples[:1])

            if callback is not None:
                callback(dps_sample=sample_img, dps_masked=composited_img)
            else:
                Image.fromarray(sample_img).save(os.path.join(directory, f"dps_sample_{i}.png"))
                Image.fromarray(composited_img).save(os.path.join(base_dir, f"{num}_dps_sample{i}_observed.png"))

        return tensor_to_np(dps_samples[:1])
