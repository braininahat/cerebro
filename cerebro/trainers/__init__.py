"""Training modules for different learning objectives.

Trainers are LightningModules that handle the training loop logic.
They are decoupled from model architectures - any compatible model
can be trained with any trainer.
"""

from .contrastive import ContrastiveFineTuner, ContrastiveTrainer
from .supervised import SupervisedTrainer
from .jepa import JEPAPhase1Trainer

__all__ = [
    "SupervisedTrainer",
    "ContrastiveTrainer",
    "ContrastiveFineTuner",
    "JEPAPhase1Trainer",
]