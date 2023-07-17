import sys
import torch
from model import PartOfSpeechEstimatorPL


model = PartOfSpeechEstimatorPL.load_from_checkpoint(sys.argv[1]).model
torch.save(model.state_dict(), sys.argv[2])
