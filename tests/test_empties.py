import torch
from PIL import Image
from Deployment.app.inference import predict


class EmptyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Allows predict() to determine the model's device.
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def forward(self, _images):
            return [{
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,)),
                "masks": torch.empty((0, 1, 16, 16)),
            }]
    
def test_empty_predictions():
    image = Image.new("RGB", (16, 16))
    model = EmptyModel()

    result = predict(image, model)

    assert result == {"predictions": []}