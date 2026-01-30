from dreamsim import dreamsim
from PIL import Image

device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)

img1 = preprocess(Image.open("/home/linyuliu/jxmount/diffusion_custom/outputs/batch_green_luxury_20260115_235452/epoch_192_best.png")).to(device)
img2 = preprocess(Image.open("/home/linyuliu/jxmount/diffusion_custom/outputs/batch_green_luxury_20260115_235452/epoch_193_best.png")).to(device)
distance = model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
print(distance)