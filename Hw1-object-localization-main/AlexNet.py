import torch.nn as nn
import torchvision.models as torchvision_models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # TODO (Q1.1): Define model
        # model definition of Alexnet - self.features defines the backbone and self.classifier defines the classifier part
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        # forward pass of AlexNet
        return self.classifier(self.features(x))


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
        bb = LocalizerAlexNet(num_classes=num_classes)
        self.features = bb.features
        self.classifier = bb.classifier
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1)

    def forward(self, x):
        # TODO (Q1.7): Define forward pass
        return self.avg_pool(self.classifier(self.features(x)))


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    model_state_dict = model.state_dict()
    if pretrained:
        print("Loading Pretrained Weights")
        alex_params = list(torchvision_models.alexnet(pretrained=True).state_dict().items())
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'features' in name:
                model_state_dict[name].copy_(alex_params[i][1])
        model.load_state_dict(model_state_dict)

    for name, param in model.named_parameters():
        if 'classifier' in name and 'weight' in name:
            nn.init.xavier_normal_(param)
        if 'classifier' in name and 'bias' in name:
            nn.init.zeros_(param)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    model_state_dict = model.state_dict()
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    if pretrained:
        alex_params = list(torchvision_models.alexnet(pretrained=True).state_dict().items())
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'features' in name:
                model_state_dict[name].copy_(alex_params[i][1])
        model.load_state_dict(model_state_dict)

    for name, param in model.named_parameters():
        if 'classifier' in name and 'weight' in name:
            nn.init.xavier_normal_(param)
        if 'classifier' in name and 'bias' in name:
            nn.init.zeros_(param)

    return model
