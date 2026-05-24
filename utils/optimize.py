import torch.optim as optim


def get_optimizer(config, parameters):
    # Defensive defaults: if config.optim missing, provide a sensible default
    if not hasattr(config, 'optim') or config.optim is None:
        # default to Adam with experiment lr/eps consistent with paper
        return optim.Adam(parameters, lr=2e-5, betas=(0.9, 0.999), eps=1e-8)

    if config.optim.optimizer == 'Adam':
        # Ensure eps is a float to guard against YAML parsing issues.
        eps = float(getattr(config.optim, 'eps', 1e-8)) if isinstance(getattr(config.optim, 'eps', 1e-8), str) else getattr(config.optim, 'eps', 1e-8)
        # allow overriding betas from config (default to 0.9,0.999)
        betas = getattr(config.optim, 'betas', (0.9, 0.999))
        # ensure tuple format
        try:
            betas = tuple(betas)
        except Exception:
            betas = (0.9, 0.999)
        weight_decay = getattr(config.optim, 'weight_decay', 0.0)
        amsgrad = getattr(config.optim, 'amsgrad', False)
        lr = float(getattr(config.optim, 'lr', 2e-5))
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay,
                          betas=betas, amsgrad=amsgrad, eps=eps)
    elif config.optim.optimizer == 'AdamW':
        # Support AdamW.
        eps = float(getattr(config.optim, 'eps', 1e-8)) if isinstance(getattr(config.optim, 'eps', 1e-8), str) else getattr(config.optim, 'eps', 1e-8)
        betas = getattr(config.optim, 'betas', [0.9, 0.999])
        lr = float(getattr(config.optim, 'lr', 2e-5))
        weight_decay = getattr(config.optim, 'weight_decay', 0.0)
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay,
                           betas=tuple(betas), eps=eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
