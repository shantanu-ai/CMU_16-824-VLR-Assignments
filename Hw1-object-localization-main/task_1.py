import argparse
import shutil
import time

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
from utils import *
from voc_dataset import *

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
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
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # the loss function in [1], is defined as Binary Cross Entropy loss using nn.BCELoss().
    # The optimizer used is Stochastic Gradient Descent with learning rate 0.01, and momentum 0.9 and
    # weight deacay 1e-4.
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
    # The dataset is created as an object of the class VOCDataset with image_size as 512.
    train_dataset = VOCDataset(split='trainval', image_size=512, top_n=30, is_collate=False)
    val_dataset = VOCDataset(split='test', image_size=512, top_n=30, is_collate=False)

    print("Finished loading dataset")
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        # collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        # collate_fn=collate_fn
    )

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="shg121_vlr-alexnet_final_robust_bb", reinit=True)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        if scheduler is not None:
            scheduler.step()
        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)
            wandb.log({"validation mean_metric1": m1})
            wandb.log({"validation mean_metric2": m2})
            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # TODO (Q1.1): Get inputs from the data dict
        # in task 1 we get the image and target classlabels from the dataloader as data['image'] and data['label']
        # respectively.
        # Convert inputs to cuda if training on GPU
        images = data['image'].to('cuda')
        target = data['label'].to('cuda')
        weights = data['wgt'].to('cuda')

        # TODO (Q1.1): Get output from model
        # the output of the model is computed using the forward method of LocalizerAlexNet or LocalizerAlexNetRobust
        # based on the argument args.arch.
        imoutput = model(images)
        if i == 0:
            print(f"Size of output: {imoutput.size()}")

        # TODO (Q1.1): Perform any necessary operations on the output
        # We are clamping the out using Maxpool and the output is also passed through sigmoid activation for the
        # BCE loss as criterion
        imoutput = clamp_out(imoutput)

        # TODO (Q1.1): Compute loss using ``criterion``
        # Compute the BCE criterion by passing the prediction as ``imoutput'' and true label as ``target''
        loss = criterion(imoutput, target)

        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        # 1. We need to reset the optimizer using optimizer.zero_grad() at every step of minibatch
        # 2. We need to backpropagate using the loss as loss.backward()
        # 3. We need to perform one step of gradient descent as optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))

            # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
            if USE_WANDB:
                i1 = epoch % args.batch_size
                i2 = (epoch + args.batch_size // 2) % args.batch_size
                im1 = wandb.Image(tensor_to_PIL(images[i1]))
                im2 = wandb.Image(tensor_to_PIL(images[i2]))

                wandb.log({
                    "train/loss": loss.item(),
                    "train/metric1": m1,
                    "train/metric2": m2,
                    "epoch": epoch,
                    "train/im1": im1,
                    "train/im2": im2,
                })
                if i == 100:
                    if epoch == 0 or epoch == 44:
                        input = torch.unsqueeze(images[0, :, :, :], 0)
                        output = model(input)
                        gt_classes = torch.where(target[0] == 1)[0][0].item()
                        plt.imsave('heat_map.png', -output[0, gt_classes, :, :].cpu().detach().numpy(), cmap='jet')
                        heat_map = Image.open('heat_map.png')
                        heat_map = heat_map.resize((512, 512))
                        wandb.log(
                            {f"train_image_epoch: {str(epoch + 1)}_iter{str(i)}_img_id_1": [
                                wandb.Image(images[0, :, :, :], caption=f"train_epoch_{epoch}_im_1")]}
                        )
                        wandb.log(
                            {f"train_heatmap_epoch: {str(epoch+1)}_iter{str(i)}_img_id_1": [
                                wandb.Image(heat_map, caption=f"train_epoch_{epoch}_hm_1")]}
                        )

                        input = torch.unsqueeze(images[1, :, :, :], 0)
                        output = model(input)
                        gt_classes = torch.where(target[1] == 1)[0][0].item()
                        plt.imsave('heat_map.png', -output[0, gt_classes, :, :].cpu().detach().numpy(), cmap='jet')
                        heat_map = Image.open('heat_map.png')
                        heat_map = heat_map.resize((512, 512))
                        wandb.log(
                            {f"train_image_epoch: {str(epoch + 1)}_iter{str(i)}_img_id_2": [
                                wandb.Image(images[1, :, :, :], caption=f"train_epoch_{epoch}_im_2")]}
                        )
                        wandb.log(
                            {f"train_heatmap_epoch: {str(epoch+1)}_iter{str(i)}_img_id_2": [
                                wandb.Image(heat_map, caption=f"train_epoch_{epoch}_im_12")]}
                        )

        # End of train()


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Got the input from val dataloader as described earlier for training
        # Convert inputs to cuda if training on GPU
        images = data['image'].cuda(0, non_blocking=True)
        target = data['label'].cuda(0, non_blocking=True)
        weights = data['wgt'].cuda(0, non_blocking=True)

        # TODO (Q1.1): Get output from model
        # Perform the forward pass for the val dataset as in training
        imoutput = model(images)

        # TODO (Q1.1): Perform any necessary functions on the output
        # Perform same clamping for the val dataset as in training
        imoutput = clamp_out(imoutput)
        # TODO (Q1.1): Compute loss using ``criterion``
        # Computed the loss for validation as in training
        loss = criterion(imoutput, target).cuda()
        # measure metrics and record loss
        m1 = metric1(imoutput.data, target)
        m2 = metric2(imoutput.data, target)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                i,
                len(val_loader),
                batch_time=batch_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))

            # TODO (Q1.3): Visualize things as mentioned in handout
            # TODO (Q1.3): Visualize at appropriate intervals
            if USE_WANDB:
                i1 = epoch % args.batch_size
                i2 = (epoch + args.batch_size // 2) % args.batch_size
                im1 = wandb.Image(tensor_to_PIL(images[i1]))
                im2 = wandb.Image(tensor_to_PIL(images[i2]))

                wandb.log({
                    "val/loss": losses.avg,
                    "val/metric1": avg_m1.avg,
                    "val/metric2": avg_m2.avg,
                    "epoch": epoch,
                    "test/im1": im1,
                    "test/im2": im2,
                })
                if i == 100:
                    if epoch == 0 or epoch == 44:
                        print("========>>> Saving images <<<<<<==============")
                        input = torch.unsqueeze(images[0, :, :, :], 0)
                        output = model(input)
                        gt_classes = torch.where(target[0] == 1)[0][0].item()
                        plt.imsave('heat_map.png', -output[0, gt_classes, :, :].cpu().detach().numpy(), cmap='jet')
                        heat_map = Image.open('heat_map.png')
                        heat_map = heat_map.resize((512, 512))
                        wandb.log(
                            {f"val_image_epoch: {str(epoch + 1)}_iter{str(i)}_img_id_1": [
                                wandb.Image(images[0, :, :, :], caption=f"val_epoch_{epoch}_im_1")]}
                        )
                        wandb.log(
                            {f"val_heatmap_epoch: {str(epoch+1)}_iter{str(i)}_img_id_1": [
                                wandb.Image(heat_map, caption=f"val_epoch_{epoch}_hm_1")]}
                        )

                        input = torch.unsqueeze(images[1, :, :, :], 0)
                        output = model(input)
                        gt_classes = torch.where(target[1] == 1)[0][0].item()
                        plt.imsave('heat_map.png', -output[0, gt_classes, :, :].cpu().detach().numpy(), cmap='jet')
                        heat_map = Image.open('heat_map.png')
                        heat_map = heat_map.resize((512, 512))
                        wandb.log(
                            {f"val_image_epoch: {str(epoch + 1)}_iter{str(i)}_img_id_2": [
                                wandb.Image(images[1, :, :, :], caption=f"val_epoch_{epoch}_im_2")]}
                        )
                        wandb.log(
                            {f"val_heatmap_epoch: {str(epoch+1)}_iter{str(i)}_img_id_2": [
                                wandb.Image(heat_map, caption=f"val_epoch_{epoch}_im_2")]}
                        )

                        input = torch.unsqueeze(images[11, :, :, :], 0)
                        output = model(input)
                        gt_classes = torch.where(target[11] == 1)[0][0].item()
                        plt.imsave('heat_map.png', -output[0, gt_classes, :, :].cpu().detach().numpy(), cmap='jet')
                        heat_map = Image.open('heat_map.png')
                        heat_map = heat_map.resize((512, 512))
                        wandb.log(
                            {f"val_image_epoch: {str(epoch + 1)}_iter{str(i)}_img_id_3": [
                                wandb.Image(images[11, :, :, :], caption=f"val_epoch_{epoch}_im_3")]}
                        )
                        wandb.log(
                            {f"val_heatmap_epoch: {str(epoch+1)}_iter{str(i)}_img_id_3": [
                                wandb.Image(heat_map, caption=f"val_epoch_{epoch}_im_3")]}
                        )

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


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


