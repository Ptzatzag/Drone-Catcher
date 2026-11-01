from PIL import Image
import torch
from .shared.shared import preprocess_image

def predict(image: Image.Image, model, threshold:float = 0.7):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    image_tensor = preprocess_image(image.convert("RGB")).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]
        
    keep = output['scores'] > threshold
    boxes = output['boxes'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    masks = output['masks'][keep].cpu().numpy()

    results = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        mask = mask[0]
        mask = (mask>0.5).astype(int)
        
        results.append({'label': int(label),
                        'score': float(score),
                        'box': box.tolist(),
                        'mask':mask.tolist()
                        })
        
    return {'predictions': results}