import os
from collections import OrderedDict

import torch.optim as optim

from model import AEModel
from utils import *


def save_plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(13, 7.5))
    ax.plot(x, y)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


def save_plot_single(x, y_16, y_128, y_1024, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(13, 7.5))
    ax.plot(x, y_16, label='Latent_size 16')
    ax.plot(x, y_128, label='Latent_size 128')
    ax.plot(x, y_1024, label='Latent_size 1024')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper left')
    plt.title(title)
    plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


def ae_loss(model, x):
    """ 
    TODO 2.1.2: fill in MSE loss between x and its reconstruction. 
    return loss, {recon_loss = loss} 
    """
    embedding = model.encoder(x)
    x_hat = model.decoder(embedding)
    bs = x.shape[0]
    loss = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(x_hat, x).view(bs, -1), dim=1))
    return loss, OrderedDict(recon_loss=loss)


def vae_loss(model, x, beta=1):
    # print(beta)
    """TODO 2.2.2 : Fill in recon_loss and kl_loss. """
    mu, log_std = model.encoder(x)
    epsilon = log_std.exp()
    z = mu + epsilon * torch.randn_like(mu)
    x_hat = model.decoder(z)
    bs = x.shape[0]
    recon_loss = torch.mean(torch.sum(torch.nn.MSELoss(reduction='none')(x_hat, x).view(bs, -1), dim=1))
    kl_loss = (-log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5).sum(1).mean()
    total_loss = recon_loss + beta * kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val=1):
    def _helper(epoch):
        return target_val

    return _helper


def linear_beta_scheduler(max_epochs=None, target_val=1):
    """TODO 2.3.2 : Fill in helper. The value returned should increase linearly 
    from 0 at epoch 0 to target_val at epoch max_epochs """
    increment_factor = target_val / (max_epochs - 1)

    def _helper(epoch):
        return increment_factor * epoch

    return _helper


def run_train_epoch(model, loss_mode, train_loader, optimizer, beta=1, grad_clip=1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)


def main(log_dir, loss_mode='vae', beta_mode='constant', num_epochs=20, batch_size=256, latent_size=256,
         target_beta_val=1, grad_clip=1, lr=1e-3, eval_interval=5):
    os.makedirs('data/' + log_dir, exist_ok=True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape=(3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    vis_x = next(iter(val_loader))[0][:36]

    # beta_mode is for part 2.3, you can ignore it for parts 2.1, 2.2
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val=target_beta_val)
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val=target_beta_val)

    epoch_list = []
    recons_list = []
    kl_list = []
    for epoch in range(num_epochs):
        print('epoch: ', epoch, 'beta: ', beta_fn(epoch))
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)

        # TODO : add plotting code for metrics (required for multiple parts)
        epoch_list.append(epoch)
        recons_list.append(val_metrics["recon_loss"])

        if (epoch + 1) % eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/' + log_dir + '/epoch_' + str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/' + log_dir + '/epoch_' + str(epoch))

            save_plot(
                epoch_list,
                recons_list,
                xlabel="Epochs",
                ylabel="Recons loss",
                title="Recons_loss vs Epochs",
                filename='data/' + log_dir + "/epoch_vs_recons",
            )

        if loss_mode == "vae":
            kl_list.append(val_metrics["kl_loss"])
            # print(recons_list)
            # print(kl_list)
            save_plot(
                epoch_list,
                kl_list,
                xlabel="Epochs",
                ylabel="KL loss",
                title="KL_loss vs Epochs",
                filename='data/' + log_dir + "/epoch_vs_kl_loss",
            )

    return epoch_list, recons_list


if __name__ == '__main__':
    # TODO: Experiments to run :
    # 2.1 - Auto-Encoder
    # Run for latent_sizes 16, 128 and 1024
    print("AE, latent: 16")
    epoch_list, recons_list_16 = main('ae_latent16', loss_mode='ae', num_epochs=20, latent_size=16)

    print("AE, latent: 128")
    epoch_list, recons_list_128 = main('ae_latent128', loss_mode='ae', num_epochs=20, latent_size=128)
    #
    print("AE, latent: 1024")
    epoch_list, recons_list_1024 = main('ae_latent1024', loss_mode='ae', num_epochs=20, latent_size=1024)

    save_plot_single(
        epoch_list,
        recons_list_16,
        recons_list_128,
        recons_list_1024,
        xlabel="Epochs",
        ylabel="Recons loss",
        title="Recons_loss vs Epochs for AE",
        filename='data/epoch_vs_recons_loss'
    )

    #
    # Q 2.2 - Variational Auto-Encoder
    print("VAE, latent: 1024")
    main('vae_latent1024', loss_mode='vae', num_epochs=20, latent_size=1024)
    #
    # # Q 2.3.1 - Beta-VAE (constant beta)
    # # Run for beta values 0.8, 1.2
    print("VAE, Beta: 0.8")
    main('vae_latent1024_beta_constant0.8', loss_mode='vae', beta_mode='constant', target_beta_val=0.8, num_epochs=20,
         latent_size=1024)

    print("VAE, Beta: 1.0")
    main('vae_latent1024_beta_constant1', loss_mode='vae', beta_mode='constant', target_beta_val=1.0, num_epochs=20,
         latent_size=1024)
    #
    print("VAE, Beta: 1.2")
    main('vae_latent1024_beta_constant1.2', loss_mode='vae', beta_mode='constant', target_beta_val=1.2, num_epochs=20,
         latent_size=1024)

    # Q 2.3.2 - VAE with annealed beta (linear schedule)
    print("VAE linear beta, Beta: 1.0")
    main(
        'vae_latent1024_beta_linear1', loss_mode='vae', beta_mode='linear',
        target_beta_val=1, num_epochs=20, latent_size=1024
    )
