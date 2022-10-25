from models.DV_model import DVModel, DVModelMINE
from models.NDT_model import NDTModel, NDTModel_iid


def build_model(config):
    if config.model_name == "dine_ndt":
        model = {'dv': DVModel(config),
                 'dv_eval': DVModel(config),
                 'ndt': NDTModel(config),
                 'ndt_eval': NDTModel(config)
                 }
    elif config.model_name == "mine_ndt":
        model = {'dv': DVModelMINE(config),
                 'dv_eval': DVModelMINE(config),
                 'ndt': NDTModel_iid(config),
                 'ndt_eval': NDTModel_iid(config)
                 }
    else:
        raise ValueError("'{}' is an invalid model name")

    return model

