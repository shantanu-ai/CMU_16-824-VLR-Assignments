import os
from glob import glob

import torch
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.utils import save_image

from utils import get_fid, interpolate_latent_space, save_plot


def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    ds_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # )
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )
        ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.

    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0, 0.9))
    scheduler_discriminator = StepLR(optim_discriminator, step_size=500000, gamma=0.1)
    optim_generator = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0, 0.9))
    scheduler_generator = StepLR(optim_generator, step_size=100000, gamma=0.1)
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
        gen,
        disc,
        num_iterations,
        batch_size,
        lamb=10,
        prefix=None,
        gen_loss_fn=None,
        disc_loss_fn=None,
        log_period=10000,
):
    gen.train()
    disc.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    print(ds_transforms)
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast():
                train_batch = train_batch.cuda()
                print("---------" * 10)
                print(iters)
                print("---------" * 10)
                real = train_batch.to(device)
                # TODO 1.2: compute generator outputs and discriminator outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                fake = gen(train_batch.size(0))
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                discrim_real = disc(real)
                discrim_fake = disc(fake)

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data
                BATCH_SIZE, C, H, W = real.shape
                epsilon = torch.rand(BATCH_SIZE, 1, 1, 1)
                epsilon = epsilon.repeat(1, C, H, W).to(device)
                interp = real * epsilon + fake * (1 - epsilon)
                discrim_interp = disc(interp)

                discriminator_loss = disc_loss_fn(
                    discrim_real, discrim_fake, discrim_interp, interp, lamb
                )
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            print(f"disc_loss: {discriminator_loss}")
            if iters % 5 == 0:
                with torch.cuda.amp.autocast():
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    fake = gen(train_batch.size(0))
                    discrim_fake = disc(fake)
                    generator_loss = gen_loss_fn(discrim_fake)
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()
                print(f"g_loss: {generator_loss}")
            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        generated_samples = gen(train_batch.size(0))
                        MEAN = torch.tensor([0.5, 0.5, 0.5]).to(device)
                        STD = torch.tensor([0.5, 0.5, 0.5]).to(device)
                        generated_samples = generated_samples * STD[:, None, None] + MEAN[:, None, None]

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    torch.jit.save(gen, prefix + "/generator.pt")
                    torch.jit.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    print(prefix)
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=32,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
    # Final FID (Full 50K): 75.32169465662764
    # Iteration 20000 FID: 58.752649720856425
    # Iteration 21000 FID: 48.24941004235552
    # Iteration 22000 FID: 50.11802109375236
    # Iteration 23000 FID: 48.05441290422118
    # Iteration 24000 FID: 46.62975289108016
    # Iteration 25000 FID: 49.663619997852095
    # Iteration 26000 FID: 70.14437839007377
    # Iteration 27000 FID: 63.04146306732463
    # Iteration 28000 FID: 51.93233016975324
    # Iteration 29000 FID: 53.72219983599973
    # Iteration 30000 FID: 64.99523568310912
