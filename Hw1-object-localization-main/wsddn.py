import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()
        self.n_channel = 256
        self.roi_dim = 15
        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, self.n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_channel, self.n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.n_channel * self.roi_dim * self.roi_dim, 4096),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
        )

        self.class_score_mlp = nn.Linear(in_features=4096, out_features=self.n_classes)
        self.bbox_score_mlp = nn.Linear(in_features=4096, out_features=self.n_classes)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):

        # TODO (Q2.1): Use image and rois as input
        roi_feat = roi_pool(
            self.features(image),
            boxes=torch.cat([torch.zeros(rois.shape[0], 1).cuda(), rois * 512], dim=-1),
            output_size=(15, 15), spatial_scale=31 / 512
        )
        flattened_feats = self.classifier(roi_feat.view(roi_feat.shape[0], -1))

        classification_scores = nn.Softmax(dim=1)(self.class_score_mlp(flattened_feats))
        detection_scores = nn.Softmax(dim=0)(self.bbox_score_mlp(flattened_feats))

        cls_prob = classification_scores * detection_scores

        if self.training:
            label_vec = gt_vec.view(self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        loss = F.binary_cross_entropy(
            torch.clamp(
                torch.sum(cls_prob, axis=0), min=0, max=1), label_vec, reduction='sum'
        )

        return loss