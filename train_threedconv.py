import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 4
learning_rate = 1e-3
epochs = 100

num_ctx_frames = 1
num_tgt_frames = 1
action_types = "flipped_jumpingjack"

model = ThreeDConv_1to9(learning_rate)
weizmann = WeizmannDataModule(batch_size, num_ctx_frames, num_tgt_frames,
                                  action_types=action_types)

logger = TensorBoardLogger('./logs', 'ThreeDConv_1to9')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger)

trainer.fit(model, weizmann)