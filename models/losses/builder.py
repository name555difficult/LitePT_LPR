from utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss

    def place_recognition_call(self, input_dict):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return None
        loss_total = 0
        stats = {}
        for c in self.criteria:
            loss, tmp = c(input_dict)
            loss_total += loss
            stats.update(tmp)
        return loss_total, stats

def build_criteria(cfg):
    return Criteria(cfg)
