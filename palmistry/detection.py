import numpy as np
from PIL import Image
import torch

def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    pil_img = Image.open(jpeg_dir)
    img = np.asarray(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2).to(device)
    pred = net(img).squeeze(0)
    pred = torch.Tensor(np.apply_along_axis(lambda x: [1,1,1] if x > 0.03 else [0,0,0], 0, pred.cpu().detach()))
    Image.fromarray((pred.permute((1,2,0)).numpy() * 255).astype(np.uint8)).save(output_dir)




import numpy as np
from PIL import Image
import torch

def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
    # Load and preprocess the image in RGB mode for faster handling
    pil_img = Image.open(jpeg_dir).convert("RGB")
    img = np.array(pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)) / 255
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Perform inference with the model
    with torch.no_grad():
        pred = net(img).squeeze(0)

    # Apply threshold and prepare the result image
    pred = (pred > 0.03).float() * 255  # Faster thresholding using PyTorch operations
    pred_img = Image.fromarray(pred.permute(1, 2, 0).byte().cpu().numpy(), mode="RGB")

    # Save the resulting image
    pred_img.save(output_dir)


# import numpy as np
# from PIL import Image
# import torch

# def detect(net, jpeg_dir, output_dir, resize_value, device=torch.device('cpu')):
#     # Load and preprocess the image in RGB mode and resize in one step
#     pil_img = Image.open(jpeg_dir).convert("RGB").resize((resize_value, resize_value), resample=Image.BILINEAR)
#     img = np.array(pil_img, dtype=np.float32) / 255  # Normalize image directly
#     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor and adjust dimensions

#     # Perform inference with the model
#     with torch.no_grad():
#         pred = net(img).squeeze(0)

#     # Apply threshold and prepare the result image
#     pred = (pred > 0.03).to(torch.uint8) * 255  # In-place thresholding and scaling
#     pred_img = Image.fromarray(pred.cpu().numpy().transpose(1, 2, 0), mode="RGB")  # Avoid permute for RGB format

#     # Save the resulting image
#     pred_img.save(output_dir)
