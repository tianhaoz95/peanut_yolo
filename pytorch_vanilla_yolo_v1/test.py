import torch
import model

yolo = model.VanillaYoloV1()
input = torch.randn(1,3,448,448)
output = yolo(input)
print(output.shape)

train_size = 20

obj_per_sample = 3

total_obj = train_size * obj_per_sample

train_input = torch.zeros(train_size,3,448,448)

y_true = {
    'class_probs': torch.randn(),
    'confs': torch.randn(),
    'coord': torch.randn(),
    'proid': torch.randn(),
    'areas': torch.randn(),
    'upleft': torch.randn(),
    'bottomright': torch.randn()
}
