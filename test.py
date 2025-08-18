import torch
from PIL import Image
import mobileclip
import cv2
import numpy as np

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='./checkpoints/mobileclip_b.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_b')

image = cv2.imread('docs/fig_accuracy_latency.png', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = image.astype(np.float32)
image /= 255.0

# c h w -> h w c
image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)


text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)