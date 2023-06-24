import torch
import os

def write_model():
    model = torch.load('models/model.pth')
    
    output_path = os.path.join('document', 'model.txt')
    
    try:
        os.makedirs('documents', exist_ok=True)
        torch.save(model, 'models/model.pth')
        with open('documents/model.txt', 'w', encoding='utf-8') as f:
                f.write(str(model))
        print('Model başarıyla kaydedildi.')
    except Exception as e:
        print('Model kaydedilirken bir hata oluştu:', str(e))

    # try:
    #     if model:
    #         with open('model.txt', 'w', encoding='utf-8') as f:
    #             f.write(str(model))
            
    # except FileExistsError and FileNotFoundError:
    #     print('The model directory does not exist')