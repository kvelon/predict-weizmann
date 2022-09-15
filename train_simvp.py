import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

from models import *
from data.data_classes import *

# Configs
batch_size = 4
learning_rate = 1e-3
epochs = 50

num_ctx_frames = 5
num_tgt_frames = 5
action_types = "flipped_jumpingjack"

hid_s=64
hid_t=256
N_s=4
N_t=8
kernel_sizes=[3,5,7,11]
groups=4

channels = 3
height = 144
width = 180
input_shape = (channels, num_ctx_frames, height, width)

model = SimVP(input_shape=input_shape, 
                   hid_s=hid_s, hid_t=hid_t, 
                   N_s=N_s, N_t=N_t,
                   kernel_sizes=kernel_sizes, 
                   groups=groups,
                   learning_rate=learning_rate)

weizmann = WeizmannDataModule(batch_size, num_ctx_frames, num_tgt_frames,
                              action_types=action_types)

logger = TensorBoardLogger('./logs', 'SimVP')

trainer = pl.Trainer(gpus=4, 
                     strategy=DDPStrategy(find_unused_parameters=False),
                     max_epochs= epochs,
                     callbacks=LearningRateMonitor(),
                     logger=logger
                     )

trainer.fit(model, weizmann)

torch.save(
    model.state_dict(),
    "./state_dicts/SimVP/experiment1"
)