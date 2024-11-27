import torch
from safetensors.torch import load_file, save_file

# Load the safetensors model
safetensors_path = './mistral/model.safetensors'
model = load_file(safetensors_path)

# Save the model in PyTorch format
pth_path = 'consolidated.0.mistral.pth'
torch.save(model, pth_path)

print(f"Model saved to {pth_path}")
