import numpy as np
import torch
import torchvision.transforms as transforms


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    bbox = bounding_boxes[confidence_score > threshold]
    scores = confidence_score[confidence_score > threshold]
    # N = bbox.shape[0]

    width_d = bbox[:, 2].reshape(bbox.shape[0], 1) - bbox[:, 0].reshape(1, bbox.shape[0])
    intersection_width = torch.minimum(width_d, torch.transpose(width_d, 0, 1))
    height_d = bbox[:, 3].reshape(bbox.shape[0], 1) - bbox[:, 1].reshape(1, bbox.shape[0])
    intersection_height = torch.minimum(height_d, torch.transpose(height_d, 0, 1))
    intersection = torch.maximum(intersection_width * intersection_height, torch.zeros_like(intersection_width))

    boxes_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    u_area = boxes_area.reshape(bbox.shape[0], 1) + boxes_area.reshape(1, bbox.shape[0]) - intersection
    iou = intersection / u_area - torch.eye(bbox.shape[0], device=bbox.device)
    maximal = torch.logical_not(
        torch.any(torch.logical_and(iou > 0.3, scores.reshape(bbox.shape[0], 1) > scores.reshape(1, bbox.shape[0])),
                  dim=0))

    return bbox[maximal], scores[maximal]


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    intersection = np.maximum(np.minimum(box1[3] - box2[1], box2[3] - box1[1]) * np.minimum(box1[3] - box2[1], box2[3] - box1[1]), 0)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    return intersection / union


def clamp_out(imoutput):
    imoutput = torch.nn.MaxPool2d(kernel_size=(imoutput.shape[2], imoutput.shape[3]))(imoutput)
    imoutput = torch.reshape(imoutput, (imoutput.shape[0], imoutput.shape[1]))
    return torch.sigmoid(imoutput)


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
        "position": {
            "minX": bbox_coordinates[i][0],
            "minY": bbox_coordinates[i][1],
            "maxX": bbox_coordinates[i][2],
            "maxY": bbox_coordinates[i][3],
        },
        "class_id": classes[i].item(),
    } for i in range(len(classes))
    ]
    # print(box_list)
    # print(type(box_list[0]["class_id"]))
    # print(type(box_list[0]["class_id"].item()))
    # print(box_list[0]["class_id"].item())
    return box_list


def get_box_data_scores(classes, bbox_coordinates, scores):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)
    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
        "position": {
            "minX": float(bbox_coordinates[i][0]),
            "minY": float(bbox_coordinates[i][1]),
            "maxX": float(bbox_coordinates[i][2]),
            "maxY": float(bbox_coordinates[i][3]),
        },
        "class_id": int(classes[i]),
        "scores": {"confidence": float(scores[i])}
    } for i in range(len(classes))
    ]

    return box_list


def log_train_wandb(USE_WANDB, wandb, iter, train_loader, data, model, id_label_map):
    if USE_WANDB and iter >= len(train_loader) - 11:
        model.eval()
        image = data['image'].cuda()
        rois = data['rois'][0].to(torch.float32).cuda()
        gt_boxes = torch.tensor(data['gt_boxes']).numpy()
        gt_class_list = torch.tensor(data['gt_classes']).numpy()

        cls_probs = model.forward(image, rois=rois)

        bb_boxes = []
        target_labels = []
        scores_labels = []
        for class_num in range(20):
            # get valid rois and cls_scores based on thresh
            confidence_score = cls_probs[:, class_num]

            # use NMS to get boxes and scores
            boxes, scores = nms(rois, confidence_score)
            bb_boxes.append(boxes.detach().cpu().numpy())
            target_labels.append([class_num for i in range(boxes.shape[0])])
            scores_labels.append(scores.detach().cpu().numpy())

        bb_boxes_np = np.concatenate(bb_boxes)
        target_labels_np = np.concatenate(target_labels).astype(int)
        scores_labels_no = np.concatenate(scores_labels)
        predictions_image = wandb.Image(tensor_to_PIL(image[0].cpu()), boxes={
            "predictions": {
                "box_data": get_box_data_scores(target_labels_np, bb_boxes_np, scores_labels_no),
                "class_labels": id_label_map,
            },
        })

        gt_image = wandb.Image(tensor_to_PIL(image[0].cpu()), boxes={
            "predictions": {
                "box_data": get_box_data(gt_class_list, gt_boxes),
                "class_labels": id_label_map,
            },
        })
        wandb.log({
            'gt_boxes': gt_image,
            'pred_boxes': predictions_image
        })


def iter_over_each_class(cls_probs, gt_class_list, total, rois, confidence, gt_boxes, correct):
    all_boxes = []
    all_labels = []
    all_scores = []

    for label in gt_class_list:
        total[label] += 1
    for class_num in range(20):
        # get valid rois and cls_scores based on thresh
        confidence_score = cls_probs[:, class_num]

        # use NMS to get boxes and scores
        boxes, scores = nms(rois, confidence_score)
        all_boxes.append(boxes.detach().cpu().numpy())
        all_labels.append([class_num for i in range(boxes.shape[0])])
        all_scores.append(scores.detach().cpu().numpy())

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

    all_boxes = np.concatenate(all_boxes)
    all_labels = np.concatenate(all_labels).astype(int)
    all_scores = np.concatenate(all_scores)

    return confidence
