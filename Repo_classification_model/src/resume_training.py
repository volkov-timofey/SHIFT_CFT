from typing import Tuple
import torch

def resume_training(
    save_path: str
  ):
  
  checkpoint = torch.load(save_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']

  return model, optimizer, epoch