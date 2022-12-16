import torch
import torch.jit as jit


class UpSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
            self,
            input_channels,
            kernel_size=3,
            n_filters=128,
            upscale_factor=2,
            padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.c_in = input_channels
        self.c_out = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.upscale_factor = upscale_factor
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
        self.conv = torch.nn.Conv2d(
            in_channels=self.c_in,
            out_channels=self.c_out,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling.
        # 1. Duplicate x channel wise upscale_factor^2 times.
        # 2. Then re-arrange to form an image of shape (batch x channel x height*upscale_factor x width*upscale_factor).
        # 3. Apply convolution.
        # Hint for 2. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle

        x = torch.cat([x, x, x, x], dim=1)
        output = x.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        upscale_factor_sq = int(self.upscale_factor ** 2)
        s_depth = int(d_depth / upscale_factor_sq)
        s_width = int(d_width * self.upscale_factor)
        s_height = int(d_height * self.upscale_factor)
        t_1 = output.reshape(batch_size, d_height, d_width, upscale_factor_sq, s_depth)
        spl = t_1.split(self.upscale_factor, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2).contiguous()

        return self.conv(output)


class DownSampleConv2D(jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
            self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.input_channels = input_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.downscale_ratio = downscale_ratio
        self.conv = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling.
        # 1. Re-arrange to form an image of shape: (batch x channel * upscale_factor^2 x height x width).
        # 2. Then split channel wise into upscale_factor^2 number of images of shape: (batch x channel x height x width).
        # 3. Average the images into one and apply convolution.
        # Hint for 1. look at
        # https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle
        output = x.permute(0, 2, 3, 1)
        downscale_ratio_sq = int(self.downscale_ratio ** 2)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * downscale_ratio_sq
        d_width = int(s_width / self.downscale_ratio)
        d_height = int(s_height / self.downscale_ratio)
        t_1 = output.split(self.downscale_ratio, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2).contiguous()
        #
        spatial_pool = output.chunk(4, dim=1)[0] + output.chunk(4, dim=1)[1] + output.chunk(4, dim=1)[2] + \
                       output.chunk(4, dim=1)[3]
        #
        # # _x = sum(output.chunk(4, dim=1)) / 4.0
        mean_spatial_pool = spatial_pool / 4.0
        # print(mean_spatial_pool.size())
        return self.conv(mean_spatial_pool)


class ResBlockUp(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.up_block = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=1),
            torch.nn.BatchNorm2d(n_filters),
            torch.nn.ReLU(),
            UpSampleConv2D(
                input_channels=n_filters, n_filters=n_filters, kernel_size=kernel_size, upscale_factor=2, padding=1
            ),
        )
        self.up_conv = UpSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1, padding=0)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        return self.up_block(x) + self.up_conv(x)


class ResBlockDown(jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
        )
        (residual): DownSampleConv2D(
            (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): DownSampleConv2D(
            (conv): Conv2d(in_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.c_in = input_channels
        self.filters = n_filters
        self.kernel_size = kernel_size
        self.down_block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            DownSampleConv2D(input_channels=n_filters, n_filters=n_filters, kernel_size=kernel_size, padding=1),

        )

        self.down_conv = DownSampleConv2D(
            input_channels=input_channels, n_filters=n_filters, kernel_size=1,
            padding=0
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Apply self.residual to the output of self.layers and apply self.shortcut to the original input.
        return self.down_block(x) + self.down_conv(x)


class ResBlock(jit.ScriptModule):
    # TODO 1.1: Implement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.res_block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        )

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        out = self.res_block(x)
        return out + x


class Generator(jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
            (layers): Sequential(
                (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (1): ReLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (4): ReLU()
        )
        (residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (shortcut): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.starting_image_size = starting_image_size
        self.n_filters = 128
        self.dense_init = torch.nn.Linear(self.n_filters, 4 * 4 * self.n_filters)
        self.gen_block = torch.nn.Sequential(
            ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=3),
            ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=3),
            ResBlockUp(input_channels=self.n_filters, n_filters=self.n_filters, kernel_size=3),
            torch.nn.BatchNorm2d(self.n_filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.n_filters, out_channels=3, kernel_size=3, padding=1),
            torch.nn.Tanh()
        )

    @jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        out = self.dense_init(z)
        out = out.reshape(-1, self.n_filters, 4, 4)
        out = self.gen_block(out)
        return out

    @jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        # Make sure to cast the latents to type half (for compatibility with torch.cuda.amp.autocast)
        z = torch.randn(n_samples, self.n_filters).to(self.device, dtype=torch.half)
        out = self.dense_init(z)
        out = out.reshape(-1, self.n_filters, 4, 4)
        out = self.gen_block(out)
        return out


class Discriminator(jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (1): ResBlockDown(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
            )
            (residual): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (shortcut): DownSampleConv2D(
                (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            )
        )
        (2): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (3): ResBlock(
            (layers): Sequential(
                (0): ReLU()
                (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (2): ReLU()
                (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        n_filters = 128
        self.dis_block = torch.nn.Sequential(
            ResBlockDown(input_channels=3, kernel_size=3, n_filters=n_filters),
            ResBlockDown(input_channels=n_filters, n_filters=n_filters),
            ResBlock(input_channels=n_filters, n_filters=n_filters, kernel_size=3),
            ResBlock(input_channels=n_filters, n_filters=n_filters, kernel_size=3),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(in_features=n_filters, out_features=1)

    @jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to flatten the output of the convolutional layers and sum across the image dimensions before
        # passing to the output layer!
        x = self.dis_block(x)
        return self.fc(torch.sum(x, dim=[2, 3]))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gen = Generator(starting_image_size=128).to(device)
    # # print(gen)
    # print(gen(32).size())
    print("----" * 10)
    inp = torch.randn(10, 3, 32, 32).to(device)
    dis = Discriminator().to(device)
    print(dis(inp).size())
