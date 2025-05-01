import torch.nn as nn


def init_model_weights(model, init_type='kaiming'):
    """
    Apply initialization to each submodule in a model.

    Args:
        model: nn.Module, the model to initialize
        init_type: str, type of initialization
            - 'kaiming': kaiming normal with relu
            - 'xavier': xavier uniform
            - 'orthogonal': orthogonal initialization
            - 'normal': normal distribution
            - 'uniform': uniform distribution
    """

    def init_func(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.parameter.Parameter)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight, -0.5, 0.5)
            else:
                raise ValueError(f'Initialization type {init_type} not supported')

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_func)

