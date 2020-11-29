import torch
import numpy as np
from tqdm import tqdm
import torch

def calculateOneSampleMetrics(target, predict, threshold=0.5, iou_threshold=0.5, n_classes=46):
    labels_target = list(target['labels'])
    labels_pred = list(predict['labels'])

    masks_target = target['masks']
    masks_pred = predict['masks'][:,0,:,:]

    masks_target = torch.reshape(masks_target > threshold, (masks_target.shape[0], -1))
    masks_pred = torch.reshape(masks_pred > threshold, (masks_pred.shape[0], -1))

    tp = prec = recall = f1 = 0
    iou = torch.tensor(0.0)
    matches = []
    matched_pred = torch.zeros(len(labels_pred))

    for i in range(len(labels_target)):
        for j in range(len(labels_pred)):
            if labels_target[i] == labels_pred[j]:

                mask_target = masks_target[i]
                mask_pred = masks_pred[j]

                area_target = torch.sum(mask_target)
                area_pred = torch.sum(mask_pred)

                intersections = torch.sum(mask_target.mul(mask_pred))
                union = area_target + area_pred - intersections
                overlaps = intersections.float() / union.float()
                if overlaps > iou_threshold:
                    matches.append((i, j, overlaps))

    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    for match in matches:
        if matched_pred[match[1]] == 0:
            matched_pred[match[1]] = 1
            iou += match[2]
            tp += 1

    if len(labels_pred) > 0:
        prec = tp / len(labels_pred)
    if len(labels_target):
        recall = tp / len(labels_target)
        m_iou = iou / len(labels_target)
    if prec > 0 and recall > 0:
        f1 = 2*prec*recall/(prec+recall)



    return [m_iou.cpu().item(), prec, recall, f1]

def calculateMetrics(test_dataloader, model, batch_size, n_classes=46, device=0):

    ious = 0
    precs = 0
    recalls = 0
    f1s = 0
    n_skip = 0
    for images, targets in tqdm(test_dataloader):

        images = list(image.to(device) for image in images)
        preds = model(images)
        # set_trace()
        # print(images.device, model.device)

        for target, pred in zip(targets, preds):
            scores = pred['scores'].cpu().numpy()
            best_idx = 0
            for _scores in scores:
                if _scores > 0.5:
                    best_idx += 1
            if best_idx == 0:
                continue
            if len(target['labels']) == 0:
                n_skip += 1
                continue
            pred = {'masks': pred['masks'][:best_idx, :, :], 'labels': pred['labels'][:best_idx]}
            target = {'masks': target['masks'].to(device), 'labels': target['labels'].to(device)}
            #print(target['image_id'])
            iou, prec, recall, f1 = calculateOneSampleMetrics(target, pred)
            # print(iou, prec, recall, f1)
            ious += iou
            precs += prec
            recalls += recall
            f1s += f1
    mean_iou = ious / (len(test_dataloader)*batch_size-n_skip)
    mean_f1 = f1s / (len(test_dataloader)*batch_size-n_skip)

    return mean_iou, mean_f1




# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')





if __name__ == '__main__':
    target_mask = np.zeros((4, 3, 3))
    target_mask[0] = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
    target_mask[1] = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]
    target_mask[2] = [[1, 1, 0],
                      [1, 1, 0],
                      [0, 0, 0]]
    target_mask[3] = [[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]

    pred_mask = np.zeros((4, 3, 3))
    pred_mask[2] = [[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]]
    pred_mask[3] = [[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]]
    pred_mask[1] = [[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]]
    pred_mask[0] = [[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1]]

    target = {'masks': torch.tensor(target_mask), 'labels': [0, 45, 3, 15]}
    pred = {'masks': torch.tensor(pred_mask), 'labels': [4, 3, 0, 17]}

    print(calculateMetrics(target, pred))