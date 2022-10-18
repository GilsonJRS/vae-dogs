import pytorch_lightning as pl
from torch import nn
import torch
import wandb
from torchvision.utils import make_grid
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import (
    resnet18_encoder, resnet18_decoder
)

class VanillaVAE(pl.LightningModule):
    def __init__(self, out_dim_encoder=512, latent_dim=256, input_height=32):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        self.fc_mu = nn.Linear(out_dim_encoder, latent_dim)
        self.fc_var = nn.Linear(out_dim_encoder, latent_dim)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean,scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1,2,3))
    
    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl =  kl.sum(-1)
        return kl
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()

        x_hat = self.decoder(z)
        
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        kl = self.kl_divergence(z,mu,std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()


        return {'elbo':elbo,
            'kl':kl.mean(),
            'recon_loss': recon_loss.mean()}

    def training_step_end(self, step_output):
        self.log_dict(step_output)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()

        x_hat = self.decoder(z)
        
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        kl = self.kl_divergence(z,mu,std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'val_elbo':elbo,
            'val_kl':kl.mean(),
            'val_recon_loss': recon_loss.mean()
        })

        return x_hat
    def validation_epoch_end(self, validation_step_outputs):

        self.logger.experiment.log({
            "val_output":[
                wandb.Image(
                    make_grid(validation_step_outputs[-1].cpu()).permute(1,2,0).numpy()
                )
            ]
        })