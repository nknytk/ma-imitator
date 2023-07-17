import torch
from transformers import AlbertConfig, AlbertModel, AlbertForMaskedLM
from transformers.models.albert.modeling_albert import AlbertMLMHead
import pytorch_lightning as pl


class PartOfSpeechEstimator(AlbertForMaskedLM):
    def __init__(self, config: AlbertConfig):
        # Call grandparents' init
        super(AlbertForMaskedLM, self).__init__(config)

        self.albert = AlbertModel(config, add_pooling_layer=False)
        _config = config.to_dict()
        _config['vocab_size'] = len(_config['parts_of_speech']) + 2
        self.predictions = AlbertMLMHead(AlbertConfig(**_config))

        # Initialize weights and apply final processing
        self.post_init()


class PartOfSpeechEstimatorPL(pl.LightningModule):
    def __init__(self, config: AlbertConfig, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = PartOfSpeechEstimator(config)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss)

    def _step(self, batch):
        output = self.model(**batch)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).logits.argmax(-1)
