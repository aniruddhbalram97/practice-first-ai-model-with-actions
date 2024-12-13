import torch
import pytest
from model import SimpleCNN
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_input():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert True
    except:
        assert False, "Model failed to process 28x28 input"

def test_model_output():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape[1] == 10, f"Model output should have 10 classes, got {output.shape[1]}"

def test_model_accuracy():
    from model import train_model
    accuracy, _ = train_model()
    assert accuracy >= 95.0, f"Model accuracy {accuracy:.2f}% is less than required 95%"

def test_relu_usage():
    model = SimpleCNN()
    relu_count = sum(isinstance(layer, torch.nn.ReLU) for layer in model.modules())
    assert relu_count > 0, "ReLU activation function is not used in the model"

if __name__ == "__main__":
    pytest.main([__file__]) 