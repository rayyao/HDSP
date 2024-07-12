import os
from ib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.train.trainers import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_functions import *
from lib.models.HDSP import build_hdsptrack
from lib.train.actors import HDSP
import importlib
from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for hdsp'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')


    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "hdsp":
        net = build_hdsptrack(cfg)
    else:
        raise ValueError("illegal script name")

    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "hdsp":
        # here cls loss and cls weight are not use
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = HDSP(net=net, objective=objective, loss_weight=loss_weight,settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    settings.save_epoch_interval = getattr(cfg.TRAIN, "SAVE_EPOCH_INTERVAL", 1)
    settings.save_last_n_epoch = getattr(cfg.TRAIN, "SAVE_LAST_N_EPOCH", 1)

    if loader_val is None:
        trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    else:
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    
    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)