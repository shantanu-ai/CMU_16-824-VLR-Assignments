import argparse
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import wandb

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from Grad_cam import GradCamModel
from utils import *
from voc_dataset import *

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet_robust')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--use-ocean', default='n', type=str)
parser.add_argument('--weighted-BCE', default='n', type=str)

best_prec1 = 0


def collate_fn(batch):
    img_arr = []
    wgt_arr = []
    rois_arr = []
    label_arr = []
    gt_boxes_arr = []
    gt_classes_arr = []

    for item in batch:
        img_arr.append(item["image"])
        label_arr.append(item["label"])
        wgt_arr.append(item["wgt"])
        rois_arr.append(item["rois"])
        gt_boxes_arr.append(item["gt_boxes"])
        gt_classes_arr.append(item["gt_classes"])

    return {
        "image": torch.stack(img_arr),
        "label": torch.stack(label_arr),
        "wgt": torch.stack(wgt_arr),
        "rois": rois_arr,
        "gt_boxes": gt_boxes_arr,
        "gt_classes": gt_classes_arr
    }


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
        for name, p, in model.named_parameters():
            print(name)
        print(torch.load("model_best.pth.tar")["state_dict"])
        model.load_state_dict(torch.load("model_best.pth.tar")["state_dict"])

        print("model loaded..")
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.load_state_dict(torch.load("model_best.pth.tar")["state_dict"])
    print("model loaded..")
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.BCELoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    print("Loading dataset")
    val_dataset = VOCDataset(split='test', image_size=512, top_n=30, is_collate=True)

    print("Finished loading dataset")
    train_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        # collate_fn=collate_fn
    )

    if args.evaluate:
        validate(val_loader, model, criterion, val_dataset.CLASS_NAMES)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="shg121_vlr-alexnet_robust-bb-gcam", reinit=True)

    validate(val_loader, model, criterion, val_dataset.CLASS_NAMES)


def validate(val_loader, model, criterion, classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    gcam = GradCamModel(model).cuda()
    # switch to evaluate mode
    model.eval()

    out_put_GT = torch.FloatTensor()
    gt = []
    end = time.time()
    for _iter, (data) in enumerate(val_loader):
        print(_iter)
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        images = data['image'].cuda(0, non_blocking=True)
        target = data['label'].cuda(0, non_blocking=True)
        gt_classes = data['gt_classes']
        weights = data['wgt'].cuda(0, non_blocking=True)

        # TODO (Q1.1): Get output from model
        imoutput = model(images)
        out, acts = gcam(images)
        acts = acts.detach().cpu()
        imoutput = clamp_out(imoutput)
        loss = criterion(imoutput, target).cuda()
        loss.backward()
        grads = gcam.get_act_grads().detach().cpu()
        pooled_grads = torch.mean(grads, dim=[0, 2, 3]).detach().cpu()
        for i in range(acts.shape[1]):
            acts[:, i, :, :] *= pooled_grads[i]

        heatmap = torch.mean(acts, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)

        heatmap = cv2.resize(heatmap.numpy(), (512, 512))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.imshow(np.array(tensor_to_PIL(images[0])))
        plt.savefig(
            os.path.join("grad_cam_op", f"img_{_iter}.png"),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close()
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        axs.imshow(np.array(heatmap))
        plt.savefig(
            os.path.join("grad_cam_op", f"hm_{_iter}.png"),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close()

        gt = torch.stack(gt_classes)
        out_put_GT = torch.cat((out_put_GT, gt.cpu()), dim=0)

        batch_time.update(time.time() - end)
        end = time.time()

    print(f"tensor cat size: {out_put_GT.size()}")

    torch.save(out_put_GT.cpu(), "gt_tensor_cat.pth.tar")


def maxPoolOutput(x):
    # print(x.size())
    output = torch.max(x, 2)
    output = torch.max(output[0], 2)
    output = output[0]

    return output


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(output, target):
    # TODO (Q1.5): compute metric1
    # nclasses = target.size(1)
    # output_cpu = output.cpu().detach().numpy()
    # target_cpu = target.cpu().detach().numpy()
    #
    # AP = []
    # for class_idx in range(nclasses):
    #     gt_cls = target_cpu[:, class_idx].astype('float32')
    #     pred_cls = output_cpu[:, class_idx].astype('float32')
    #     if np.count_nonzero(gt_cls) != 0:
    #         pred_cls -= 1e-5 * gt_cls
    #         ap = sklearn.metrics.average_precision_score(gt_cls, 1e-5 * gt_cls)
    #         AP.append(ap)
    # return np.mean(AP)
    nclasses = target.size(1)
    AP = []
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    for class_idx in range(nclasses):
        gt_cls = target[:, class_idx].astype('float32')
        if np.sum(gt_cls) == 0:
            continue
        AP.append(
            sklearn.metrics.average_precision_score(
                y_true=gt_cls,
                y_score=output[:, class_idx].astype('float32')
            )
        )
    return np.mean(AP)


def metric2(output, target):
    recall = sklearn.metrics.recall_score(
        target.cpu().detach().numpy().astype('float32'),
        output.cpu().detach().numpy().astype('float32') > 0.5, average='samples'
    )
    return recall


if __name__ == '__main__':
    main()
