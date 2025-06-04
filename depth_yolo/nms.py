import torch
import torchvision

def xywh2xyxy(x):
    # x: (n,4) tensor [x_center, y_center, w, h]
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    max_det: int = 300,
):
    """
    Упрощённый NMS, возвращает list длины batch, каждый элемент — tensor (m,6)
    [x1, y1, x2, y2, conf, cls]
    """
    # 1) если передали tuple/list, берём первый элемент
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs, _, _ = prediction.shape  # batch size
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for i in range(bs):
        x = prediction[i]
        # 2) фильтрация по obj_conf > conf_thres
        mask = x[:, 4] > conf_thres
        x = x[mask]
        if not x.shape[0]:
            continue

        # 3) умножаем cls_conf на obj_conf
        x[:, 5:] *= x[:, 4:5]

        # 4) преобразуем боксы xywh→xyxy
        box = xywh2xyxy(x[:, :4])

        # 5) выбираем для каждого бокса лучшую метку
        conf, cls = x[:, 5:].max(1, keepdim=True)
        det = torch.cat((box, conf, cls.float()), dim=1)
        # 6) ещё раз фильтрация по conf
        det = det[det[:, 4] > conf_thres]
        if not det.shape[0]:
            continue

        # 7) фильтрация по классам (если нужно)
        if classes is not None:
            det = det[(det[:, 5:6] == torch.tensor(classes, device=det.device)).any(1)]
            if not det.shape[0]:
                continue

        # 8) сортировка по confidence и отсечение лишних
        det = det[det[:, 4].argsort(descending=True)[: max_det * 10]]

        # 9) batched NMS
        c = det[:, 5:6] * (0 if agnostic else 4096)  # сдвиг по классам
        boxes, scores = det[:, :4] + c, det[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]
        output[i] = det[keep]

    return output
