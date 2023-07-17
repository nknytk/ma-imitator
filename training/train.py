import json
import sys
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AlbertConfig
from model import PartOfSpeechEstimatorPL
from custom_dataset import CustomDataset

BATCH_SIZE = 64
MAX_EPOCH = 40
ACCUMULATE_GRAD_BATCHES = [1, 4, 16]


def train(config_file_path: str, checkpoint_path: str = None):
    config_name = config_file_path.split('.')[0].split('/')[-1]
    config = AlbertConfig.from_json_file(config_file_path)
    del(config.chars)
    train_dataset = CustomDataset('data/encoded/train.jsonl', config.max_position_embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CustomDataset('data/encoded/val.jsonl', config.max_position_embeddings)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    save_model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath=f'models/{config_name}'
    )
    early_stopping_checkpoint = pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=1)

    for accumulate_grad_batches in ACCUMULATE_GRAD_BATCHES:
        model = PartOfSpeechEstimatorPL.load_from_checkpoint(checkpoint_path) if checkpoint_path else PartOfSpeechEstimatorPL(config)
        trainer = pl.Trainer(
            devices=1,
            accelerator='gpu',
            max_epochs=MAX_EPOCH,
            callbacks=[save_model_checkpoint, early_stopping_checkpoint],
            accumulate_grad_batches=accumulate_grad_batches
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        print(save_model_checkpoint.best_model_path)
        checkpoint_path = save_model_checkpoint.best_model_path


if __name__ == '__main__':
    if len(sys.argv) > 2:
        train(sys.argv[1], sys.argv[2])
    else:
        train(sys.argv[1])
