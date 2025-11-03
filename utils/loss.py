# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    创新亮点四：引入"知识蒸馏"提升小模型性能.

    计算学生模型和教师模型输出之间的蒸馏损失，使用KL散度来衡量两个概率分布的差异。
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        """
        初始化蒸馏损失函数.

        Args:
            temperature (float): 温度参数，用于软化概率分布
            alpha (float): 蒸馏损失的权重，范围[0,1]
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs, targets, hard_loss):
        """
        计算蒸馏损失.

        Args:
            student_outputs: 学生模型的输出 [batch_size, anchors, grid, grid, classes+5]
            teacher_outputs: 教师模型的输出 [batch_size, anchors, grid, grid, classes+5]
            targets: 真实标签
            hard_loss: 原始的hard loss

        Returns:
            total_loss: 总损失 = (1-alpha) * hard_loss + alpha * soft_loss
        """
        soft_loss = 0.0

        # 对每个检测层计算蒸馏损失
        for i, (student_out, teacher_out) in enumerate(zip(student_outputs, teacher_outputs)):
            # YOLO输出格式: [batch, anchors, grid_y, grid_x, classes+5]
            # 提取分类预测部分 (去掉坐标和置信度)
            student_cls = student_out[..., 5:]  # [batch, anchors, grid_y, grid_x, student_classes]
            teacher_cls = teacher_out[..., 5:]  # [batch, anchors, grid_y, grid_x, teacher_classes]

            # 检查特征图大小是否匹配 (grid_y, grid_x)
            student_shape = student_cls.shape[2:4]  # [grid_y, grid_x]
            teacher_shape = teacher_cls.shape[2:4]  # [grid_y, grid_x]

            if student_shape != teacher_shape:
                # 如果特征图大小不匹配，使用双线性插值调整教师模型输出
                # 重塑为 [batch*anchors, classes, grid_y, grid_x] 进行插值
                batch, anchors, _, _, classes = teacher_cls.shape
                teacher_cls_reshaped = teacher_cls.permute(
                    0, 1, 4, 2, 3
                ).contiguous()  # [batch, anchors, classes, grid_y, grid_x]
                teacher_cls_reshaped = teacher_cls_reshaped.view(
                    batch * anchors, classes, teacher_shape[0], teacher_shape[1]
                )

                # 插值到学生模型的特征图大小
                teacher_cls_reshaped = torch.nn.functional.interpolate(
                    teacher_cls_reshaped, size=student_shape, mode="bilinear", align_corners=False
                )

                # 恢复原始形状
                teacher_cls = teacher_cls_reshaped.view(batch, anchors, classes, student_shape[0], student_shape[1])
                teacher_cls = teacher_cls.permute(
                    0, 1, 3, 4, 2
                ).contiguous()  # [batch, anchors, grid_y, grid_x, classes]

            # 处理类别数不匹配的情况
            student_nc = student_cls.shape[-1]
            teacher_nc = teacher_cls.shape[-1]

            if student_nc != teacher_nc:
                # 如果类别数不匹配，只使用较小的类别数进行蒸馏
                min_nc = min(student_nc, teacher_nc)
                student_cls = student_cls[..., :min_nc]
                teacher_cls = teacher_cls[..., :min_nc]

            # 重塑张量为 [batch*anchors*grid_y*grid_x, classes] 以便计算KL散度
            student_cls_flat = student_cls.contiguous().view(-1, student_cls.shape[-1])
            teacher_cls_flat = teacher_cls.contiguous().view(-1, teacher_cls.shape[-1])

            # 应用温度软化
            student_soft = torch.log_softmax(student_cls_flat / self.temperature, dim=-1)
            teacher_soft = torch.softmax(teacher_cls_flat / self.temperature, dim=-1)

            # 计算KL散度
            kl_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature**2)
            soft_loss += kl_loss

        # 计算总损失
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        return total_loss, soft_loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, opt=None, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        # Box loss type selection
        self.box_loss = getattr(opt, "box_loss", "ciou").lower() if opt else "ciou"

        # 知识蒸馏相关参数
        self.distillation = getattr(opt, "distillation", False) if opt else False
        if self.distillation:
            self.distill_loss = DistillationLoss(
                temperature=getattr(opt, "distill_temp", 4.0), alpha=getattr(opt, "distill_alpha", 0.7)
            )

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # Bbox loss - choose between CIoU and WIoU
                if self.box_loss == "wiou":
                    loss_wiou = WIoU(pbox, tbox[i])
                    lbox += loss_wiou.wiou
                    # For objectness loss, we still need IoU values
                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                elif self.box_loss == "ciou":
                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss
                else:
                    # Default to CIoU if unknown loss type
                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def __call_with_distillation__(self, student_pred, teacher_pred, targets):
        """
        计算包含知识蒸馏的损失.

        Args:
            student_pred: 学生模型预测
            teacher_pred: 教师模型预测
            targets: 真实标签

        Returns:
            total_loss: 总损失
            loss_items: 损失项详情 [hard_loss, soft_loss, total_loss]
        """
        # 计算原始hard loss
        hard_loss, hard_loss_items = self.__call__(student_pred, targets)

        if self.distillation and teacher_pred is not None:
            # 计算蒸馏损失
            total_loss, soft_loss = self.distill_loss(student_pred, teacher_pred, targets, hard_loss)

            # 返回详细的损失信息
            loss_items = torch.cat(
                [
                    hard_loss_items,  # [lbox, lobj, lcls]
                    torch.tensor([soft_loss], device=self.device),  # soft loss
                    torch.tensor([total_loss], device=self.device),  # total loss
                ]
            ).detach()

            return total_loss, loss_items
        else:
            # 没有蒸馏时，返回原始损失
            return hard_loss, hard_loss_items

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


# =====================================================================================
# 以下为WIoU损失函数代码
# =====================================================================================


class WIoU:
    """
    Wise-IoU loss function.
    https://arxiv.org/abs/2301.10051.
    """

    def __init__(self, pred, target, eps=1e-7, alpha=2.0, beta=4.0):
        """Initialize WIoU loss with prediction and target boxes."""
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.pred = pred
        self.target = target
        # Calculate basic IoU first
        self.iou = bbox_iou(pred, target, xywh=True, CIoU=False).squeeze()

    @property
    def wiou(self):
        """Calculate WIoU loss according to the paper formula."""
        # Ensure pred and target have same shape
        pred = self.pred
        target = self.target

        # Calculate the distance between the center points of the two bounding boxes
        dist = torch.sum((pred[:, :2] - target[:, :2]) ** 2, dim=1)

        # Convert from center format (x, y, w, h) to corner format for enclosing box calculation
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        target_x1 = target[:, 0] - target[:, 2] / 2
        target_y1 = target[:, 1] - target[:, 3] / 2
        target_x2 = target[:, 0] + target[:, 2] / 2
        target_y2 = target[:, 1] + target[:, 3] / 2

        # Enclosing box dimensions
        cw = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        ch = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)

        # R_WIoU calculation according to paper formula (3.14)
        r_wiou = torch.exp(dist / (cw**2 + ch**2 + self.eps))

        # Final WIoU loss calculation according to paper formula (3.13)
        # Use a detachable beta to construct the focusing factor
        beta = (self.iou.detach() / self.alpha).pow(self.beta)
        loss_wiou = r_wiou * (1 - self.iou) * beta
        return loss_wiou.mean()


# =====================================================================================
# WIoU损失函数代码结束
# =====================================================================================
