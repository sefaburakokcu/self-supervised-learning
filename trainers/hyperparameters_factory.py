import torch.optim as optim


class OptimizerFactory:
    """Factory for creating optimizers"""

    @staticmethod
    def create(optimizer_name: str, params, optimizer_params: dict) -> optim.Optimizer:
        """Create optimizer from name and parameters"""

        if optimizer_name.lower() == 'sgd':
            return optim.SGD(params, **optimizer_params)

        elif optimizer_name.lower() == 'adam':
            return optim.Adam(params, **optimizer_params)

        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(params, **optimizer_params)

        elif optimizer_name.lower() == 'rmsprop':
            return optim.RMSprop(params, **optimizer_params)

        elif optimizer_name.lower() == 'adadelta':
            return optim.Adadelta(params, **optimizer_params)

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers"""

    @staticmethod
    def create(scheduler_name: str, optimizer: optim.Optimizer,
               scheduler_params: dict, epochs: int = None) -> object:
        """Create scheduler from name and parameters"""

        if scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

        elif scheduler_name.lower() == 'exponential':
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)

        elif scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)

        elif scheduler_name.lower() == 'cyclical':
            return optim.lr_scheduler.CyclicLR(optimizer, **scheduler_params)

        elif scheduler_name.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

        elif scheduler_name.lower() == 'warmup_cosine':
            # Custom warmup + cosine
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)

        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")