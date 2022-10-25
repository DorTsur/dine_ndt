from trainers.trainer import build_trainer
from data.data_loader import load_data
from models.models import build_model
from utils.utils import preprocess_meta_data

import logging
import os
import sys

logger = logging.getLogger("logger")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = preprocess_meta_data()

    # define data loader
    data = load_data(config)

    if not config.quiet:
        config.print()

    # create a model
    model = build_model(config)

    # create trainer and pass all the previous components to it
    trainer = build_trainer(model, data, config)

    # train the model
    trainer.train()

    trainer.final_eval()

if __name__ == '__main__':
    main()