import cv2
import pytesseract
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from models.write import write_model 

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

image_path = 'images/receipt2.jpg'
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image_rgb).unsqueeze(0)

model = resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model_path = 'models/model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

if predicted_class == 1:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(gray, lang='tur', config=custom_config)

    print(text)

    lines = text.split('\n')
    total_amount = None

    for line in lines:
        if 'TOPLAM' in line or 'TUTAR' in line:
            words = line.split()
            for word in words:
                if word.isdigit():
                    total_amount = float(word)
                    break

    if total_amount is not None:
        print('Total Amount:', total_amount)
    else:
        print('Toplam Tutar Bulunamadı')

    try:
        with open('documents/receipt.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    except FileNotFoundError:
        print("The 'docs' directory does not exist")
else:
    print('Görüntüde fiş bulunamadı.')

model = torch.load('models/model.pth')

try:
    if model:
        with open('documents/model.txt', 'w', encoding='utf-8') as f:
            f.write(str(model))
            
except FileExistsError and FileNotFoundError:
    print('The model directory does not exist')

write_model()

# import cv2
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# image_path = 'images/fis2.jpg'

# image = cv2.imread(image_path)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray = cv2.medianBlur(gray, 3)

# custom_config = r'--oem 3 --psm 6'
# text = pytesseract.image_to_string(gray, lang='tur', config=custom_config)

# print(text)

# lines = text.split('\n')
# total_amount = None

# for line in lines:
#     if 'TOPLAM' in line or 'TUTAR' in line:
#         words = line.split()
#         for word in words:
#             if word.isdigit():
#                 total_amount = float(word)
#                 break

# if total_amount is not None:
#     print('Total Amount in Receipt:', total_amount)
# else:
#     print('Total Amound Not Found')

# try:
#     with open('receipt.txt', 'w', encoding='utf-8') as f:
#         f.write(text)
# except FileNotFoundError:
#     print("The 'docs' directory does not exist")








# import cv2
# import pytesseract 

# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# image_path = 'images/fis.jpg'

# image = cv2.imread(image_path)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray = cv2.medianBlur(gray, 3)

# # text = pytesseract.image_to_string(gray, lang='eng')
# text = pytesseract.image_to_string(gray, lang='tur')

# print(text)

# lines = text.split('\n')
# total_amount = None

# for line in lines:
#     if 'TOPLAM' in line or 'TUTAR' in line:
#         words = line.split()
#         for word in words:
#             if word.isdigit():
#                 total_amount = float(word)
#                 break

# if total_amount is not None:
#     print('Fisdeki Toplam Tutar:', total_amount)
# else:
#     print('Toplam Tutar Bulunamadı')


# try:
#     with open('receipt.txt', 'w', encoding='utf-8') as f:
#         f.write(text)
# except FileNotFoundError:
#     print("The 'docs' directory does not exist")