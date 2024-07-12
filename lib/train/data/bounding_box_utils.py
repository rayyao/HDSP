import torch
def rect_to_rel(bb, sz_norm=None):
    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


def rel_to_rect(bb, sz_norm=None):


    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def masks_to_bboxes(mask, fmt='c'):

    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = m.sum(dim=-2).nonzero()
        my = m.sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)


def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes
