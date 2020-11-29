import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN, MaskRCNNHeads
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.roi_heads import RoIHeads, maskrcnn_inference, maskrcnn_loss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

from collections import OrderedDict
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import warnings

def get_model(pretrained=False, progress=True, nr_class=91, attr_score_thresh=0.7, pos_weight=100., pretrained_backbone=True, **kwargs):
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    if pretrained:
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = TripleMaskRCNN(backbone, nr_class, attr_score_thresh=attr_score_thresh, pos_weight=pos_weight, **kwargs)

    # TODO: check whether use strict=False?
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        msg = model.load_state_dict(state_dict)
    return model

class TripleRoIHeads(RoIHeads):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 # Attribute
                 attr_score_thresh=0.7,
                 pos_weight=100.
                 ):
        super(TripleRoIHeads, self).__init__(
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
        )
        # self.bce = nn.BCEWithLogitsLoss()
        self.attr_score_thresh = attr_score_thresh
        self.pos_weight = pos_weight

    def postprocess_detections(self, class_logits, attr_logits, box_regression, proposals, image_shapes):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        # class_logits: (1000,47), attr-logits: (1000,295)
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        num_attrs = attr_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_ascores = F.sigmoid(attr_logits)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
            pred_ascores_list = [pred_ascores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0) # (bs,1000,47,4)
            pred_scores_list = pred_scores.split(boxes_per_image, 0) # (bs, 1000, 47)
            pred_ascores_list = pred_ascores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_attrs = []
        for boxes, scores, ascores, image_shape in zip(pred_boxes_list, pred_scores_list, pred_ascores_list, image_shapes): # for each image
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores) # (1000,47) 都是从0,46结束

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            ascores = ascores[:, 1:] # (1000,294)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4) # (46000,4)
            scores = scores.reshape(-1) # (46000,)
            labels = labels.reshape(-1) # (46000,)
            ascores = ascores.unsqueeze(0).repeat(46,1,1).reshape(-1, num_attrs-1) # (46,1000,294)->(46000,294)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1) # 压缩成1d的，用thresh去卡
            boxes, scores, labels, ascores = boxes[inds], scores[inds], labels[inds], ascores[inds]

            # remove empty boxes
            # attention!!! 压缩成1d的，用thresh去卡（每个box会预测46个类别的box，所以一个box对一个46个box，因此卡thresh是对类别单位的box进行的）
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, ascores = boxes[keep], scores[keep], labels[keep], ascores[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, ascores = boxes[keep], scores[keep], labels[keep], ascores[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

            # ascores: (post_rois, 294)
            # brute force
            ascore_list = []
            for ascore in ascores:
                ascore_list.append(torch.where(ascore > self.attr_score_thresh)[0]) # list(Tensor()) # 长度不均匀的一维tensor
            all_attrs.append(ascore_list) # list[(post_rois, 294)]

        # all_scores: list[(post_rois,)], all_labels: list[(post_rois,)], all_attrs: list[list[attr1,attr2], (...)] 每个list里有每个roi的attr tensor
        return all_boxes, all_scores, all_labels, all_attrs 

    def triplercnn_loss(self, class_logits, box_regression, labels, regression_targets,
        attr_logits, attr_labels, pweight):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor])
        """
        Computes the loss for Faster R-CNN.

        Arguments:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)
            attr_labels (Tensor)
        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0) # concat on batch size dimension
        attr_labels = torch.cat(attr_labels, dim=0).to(torch.float32)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)
        # print(attr_logits.shape, attr_labels[0].shape)
        # attribute_loss = self.bce(attr_logits, attr_labels) # (1024, 295), attr_labels: (1024, 295)
        
        # reset BCE pos weight for each batch
        pos_weight = torch.ones_like(attr_labels).cuda()
        pos_weight[attr_labels.bool()] = pweight
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
        attribute_loss = bce(attr_logits, attr_labels) # (1024, 295), attr_labels: (1024, 295)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss, attribute_loss

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets, attrs = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        attr_logits = self.attr_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # labels: 每个proposal和gt匹配后分配得到的class id
            # class_logits: num_class-d的输出
            # loss_classifier, loss_box_reg = fastrcnn_loss(
            #     class_logits, box_regression, labels, regression_targets)
            loss_classifier, loss_box_reg, loss_attribute = self.triplercnn_loss(
                class_logits, box_regression, labels, regression_targets, attr_logits, attrs, self.pos_weight)

            # attribute loss: attribute-wise BCE
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_attribute": loss_attribute,
            }
        else:
            boxes, scores, labels, ascores = self.postprocess_detections(class_logits, attr_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            # set_trace()
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "ascores": ascores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result] # predict box
            # print(mask_proposals) # 在训练的时候全部为空
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals) # batch size
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1) # predict box中class id为非背景类的idx
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos]) # batchsize中为非背景的proposal所分配道的gt box的idx
                # print(mask_proposals[0].shape, mask_proposals[1].shape)
                # [(nr_roi1, 4), ...]
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets] # [(nr_objs,), (nr_objs2,)]
                # batch_size, (nr_objs, 800, 800), (nr_objs2, 800, 800), (X, 47, 28, 28) X: 2个图片的Roi数目
                # print(len(gt_masks), gt_masks[0].shape, gt_masks[1].shape, mask_logits.shape) 
                # print(gt_labels[0].shape, gt_labels[1].shape)
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {
                    "loss_mask": rcnn_loss_mask
                }
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if self.keypoint_roi_pool is not None and self.keypoint_head is not None \
                and self.keypoint_predictor is not None:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = {
                    "loss_keypoint": rcnn_loss_keypoint
                }
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, attr_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        matched_idxs = []
        labels = []
        attrs = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, attr_labels_in_image in zip(proposals, gt_boxes, gt_labels, attr_labels):
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix) # 返回每个proposal匹配得到的最佳gt box的idx

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image] # IOU>0.5的proposal为fg，标签为对应的class id
            attrs_in_image = attr_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)
            attrs_in_image = attrs_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = torch.tensor(0) # IOU<0.5的则为bg id
            # print(torch.tensor(0).device, labels_in_image.device, attrs_in_image.device)
            attrs_in_image[bg_inds] = torch.zeros((295,), dtype=torch.int64).cuda()

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler
            # TODO: 如何介于两个thresh之间，attribute的loss如何计算，先统一成全部都是0吧
            attrs_in_image[ignore_inds] = torch.zeros((295,), dtype=torch.int64).cuda()

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            attrs.append(attrs_in_image)

        return matched_idxs, labels, attrs

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        attr_labels = [t["attrs"] for t in targets] # TODO

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, attrs = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, attr_labels) # matched_idxs: 每个proposal匹配得到的最佳的gt box idx, labels: 该proposal应分配得到的class id 
        
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals) # batch size
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])
            attrs[img_id] = attrs[img_id][img_sampled_inds]

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, attrs

class TripleFasterRCNN(GeneralizedRCNN):
    """
    slightly modify from Faster RCNN
    - use TripleRoIHeads
    """
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None, attr_score_thresh=0.7, pos_weight=100.):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = TripleRoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            attr_score_thresh=attr_score_thresh,
            pos_weight=pos_weight)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(TripleFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TripleMaskRCNN(TripleFasterRCNN):
    """
    slightly modify from MaskRCNN
    - add Attribute Head
    """
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 # Attribute parameters
                 attr_score_thresh=0.7, pos_weight=100.):

        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)

        super(TripleMaskRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights, attr_score_thresh, pos_weight)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

        # modify @zhengkai
        class AttrPredictor(nn.Module):
            def __init__(self, in_channels=1024, num_classes=295):
                super(AttrPredictor, self).__init__()
                self.cls_score = nn.Linear(in_channels, num_classes)

            def forward(self, x):
                if x.dim() == 4:
                    assert list(x.shape[2:]) == [1, 1]
                x = x.flatten(start_dim=1)
                scores = self.cls_score(x)

                return scores

        self.roi_heads.attr_predictor = AttrPredictor(num_classes=295)


if __name__ == '__main__':
    from ipdb import set_trace
    model = get_model(nr_class=47)
    model.eval()
    print(model)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    set_trace()
    print("finish...")
    