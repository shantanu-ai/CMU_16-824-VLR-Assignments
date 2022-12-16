from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle as pkl

import torch.utils.model_zoo as model_zoo
import wandb
from torch.nn.parameter import Parameter

from task_1 import AverageMeter
from utils import log_train_wandb, nms, iou
from voc_dataset import *
# imports
from wsddn import WSDDN

# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    help='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    help='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float,
    help='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    help='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=500,
    type=int,
    help='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    help='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    help='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

USE_WANDB = True
if USE_WANDB:
    wandb.init(project="shg121-vlr1q2-1")


def calculate_map(correct, confidence, total):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    APs = []
    for class_num in range(20):
        class_correct = np.array(correct[class_num])
        if len(confidence[class_num]) > 0:
            cls_scores = []
            for i in confidence[class_num]:
                cls_scores.append(i.item())
            class_scores = np.array(cls_scores)
        else:
            class_scores = np.array(confidence[class_num])

        class_total = total[class_num]

        indices = np.argsort(-1 * np.array(class_scores))
        class_correct = class_correct[indices]
        precisions = []
        recalls = []
        tp = 0
        fp = 0
        for i in range(len(class_correct)):
            if class_correct[i]:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / class_total)

        AP = 0
        r1 = 0
        for i in range(1, len(precisions)):
            r2 = recalls[i]
            if r1 < r2:
                AP += (r2 - r1) * precisions[i]
                r1 = r2
        APs.append(AP)
    return APs


def test_model(model, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    correct = [[] for i in range(20)]
    confidence = [[] for i in range(20)]
    total = [0 for i in range(20)]

    for iter, data in enumerate(val_loader):
        # if iter == 10:
        #            break
        # one batch = data for one image
        image = data['image'].cuda()
        rois = data['rois'][0].to(torch.float32).cuda()
        gt_boxes = torch.tensor(data['gt_boxes']).numpy()
        gt_class_list = torch.tensor(data['gt_classes']).numpy()

        # TODO: perform forward pass, compute cls_probs
        cls_probs = model.forward(image, rois=rois)

        for label in gt_class_list:
            total[label] += 1

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):
            # get valid rois and cls_scores based on thresh
            confidence_score = cls_probs[:, class_num]

            # use NMS to get boxes and scores
            boxes, scores = nms(rois, confidence_score)
            for i in range(boxes.shape[0]):
                box = boxes[i]
                score = scores[i]
                confidence[class_num].append(score)
                box_correct = False
                if class_num in gt_class_list:
                    for j in range(gt_class_list.shape[0]):
                        if gt_class_list[j] == class_num and iou(gt_boxes[j], box.detach().cpu().numpy()) > 0.5:
                            box_correct = True

                correct[class_num].append(box_correct)

    # TODO: visualize bounding box predictions when required

    # TODO: Calculate mAP on test set
    APs = calculate_map(correct, confidence, total)
    return APs


def train_model(model, class_names, train_loader=None, val_loader=None, optimizer=None, args=None, id_label_map=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = AverageMeter()
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):
            print(f"iter: {iter}")
            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            #
            # # TODO (Q2.2): perform forward pass
            # # take care that proposal values should be in pixels
            # # Convert inputs to cuda if training on GPU
            image = image.cuda()
            wgt = wgt.cuda()
            rois = rois[0].to(torch.float32).cuda()
            target = target.cuda()

            # start
            # TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
            # also convert inputs to cuda if training on GPU
            model.forward(image, rois=rois, gt_vec=target)
            # end

            # backward pass and update
            loss = model.loss
            train_loss.update(loss.item())
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                with torch.no_grad():
                    ap = test_model(model, val_loader)
                print(f"==> AP:  {ap} || mAP: {np.mean(ap)} || train_loss: {train_loss.avg}")
                if USE_WANDB:
                    wandb.log({**{
                        "test/" + class_names[i] + "_ap": ap[i] for i in range(20)
                    }, "test/mAP": np.mean(ap)})
                model.train()

            if iter % 500 == 0:
                wandb.log({"train/loss": train_loss.avg})
                train_loss.reset()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            log_train_wandb(USE_WANDB, wandb, iter, train_loader, data, model, id_label_map)
            model.train()


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    train_dataset = VOCDataset(split='trainval', image_size=512, top_n=300, is_collate=True)
    val_dataset = VOCDataset(split='test', image_size=512, top_n=300, is_collate=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # batchsize is one for this implementation
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
                 open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    print(net)
    # for child in net.features.children():
    #     for param in child.parameters():
    #         param.requires_grad = False
    #
    # for child in net.classifier.children():
    #     for param in child.parameters():
    #         param.requires_grad = False

    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    id_label_map = dict(enumerate(val_dataset.CLASS_NAMES))
    # Training
    train_model(net, train_dataset.CLASS_NAMES, train_loader, val_loader, optimizer, args, id_label_map)


if __name__ == '__main__':
    main()
