def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

import torch


def MyOptimizers(self):
    optimizer = self.args.optimizer(self.parameters(), lr=self.args.lr)
    if self.args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min', factor=0.5,
                                                               patience=5, verbose=True,
                                                               threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=0,
                                                               eps=1e-07)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'training_loss'}
    elif self.args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
    # elif self.args.scheduler == 'MultiStepLR':
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.5)
    elif self.args.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return [optimizer], [scheduler]
    elif self.args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return [optimizer], [scheduler]
    elif self.args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)
        return [optimizer], [scheduler]
    else:
        scheduler = None
        return [optimizer], [scheduler]

