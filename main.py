import torch

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS is available. GPU can be used for training.")
    device = torch.device("mps")
else:
    print("MPS is not available. Training will be done on CPU.")
    device = torch.device("cpu")

# Example usage
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(x)