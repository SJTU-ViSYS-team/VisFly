import torch
import torch.nn as nn

def check_none_parameters(model):
    for name, param in model.named_parameters():
        if param is None:
            print(f"Uninitialized parameter found in layer: {name}")

def get_network_statistics(model, logger,is_record):
    stats = {}

    if is_record:
        for name, param in model.named_parameters():
            key = "debug/" + name.replace("extractor", "EX").replace("feature","FR").replace("weight", "w").replace("bias","b")
            if 'weight' in name:
                stats = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'max': param.data.abs().max().item(),
                }
                logger.record(key+".mean", stats["mean"])
                logger.record(key+".std", stats["std"])
                logger.record(key+".max", stats["max"])

            logger.record(key, param)





    # return stats

