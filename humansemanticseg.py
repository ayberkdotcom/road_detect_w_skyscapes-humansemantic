import cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import matplotlib.pyplot as plt

device = 'cpu'
deeplab_model = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
deeplab_model.eval()
deeplab_model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video akışı kapandı.")
        break

    original_height, original_width = frame.shape[:2]

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = transform(image_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = deeplab_model(input_image)['out'][0]
    pred = torch.argmax(pred, dim=0).cpu().numpy()  
    
    
    mask = np.where(pred == 15, 1, 0)  

    mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    colored_image = frame.copy()
    colored_image[mask_resized == 1] = [0, 255, 0]  

    
    cv2.imshow('Segmented Video', colored_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
