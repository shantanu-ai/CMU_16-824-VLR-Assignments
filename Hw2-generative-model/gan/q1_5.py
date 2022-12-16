import os

import torch

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
        discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    # TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    # loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    gradient = torch.autograd.grad(
        inputs=interp,
        outputs=discrim_interp,
        grad_outputs=torch.ones_like(discrim_interp),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    # print(gradient_norm.shape)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    d_loss = discrim_fake.mean() - discrim_real.mean() + lamb * gradient_penalty
    return d_loss


def compute_generator_loss(discrim_fake):
    # TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = -discrim_fake.mean()
    return loss


if __name__ == "__main__":
    gen = Generator().cuda().to(memory_format=torch.channels_last)
    disc = Discriminator().cuda().to(memory_format=torch.channels_last)
    prefix = "data_wgan_gp/"

    print(prefix)
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
    )
