import os
import argparse
from tools import *
from model import *
from rectification import *
from detection import *
from classification import *
from measurement import *

def main(input):
    path_to_input_image = 'input/{}'.format(input)

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    resize_value = 256
    path_to_clean_image = 'results/palm_without_background.jpg'
    path_to_warped_image = 'results/warped_palm.jpg'
    path_to_warped_image_clean = 'results/warped_palm_clean.jpg'
    path_to_warped_image_mini = 'results/warped_palm_mini.jpg'
    path_to_warped_image_clean_mini = 'results/warped_palm_clean_mini.jpg'
    path_to_palmline_image = 'results/palm_lines.png'
    path_to_model = 'checkpoint/checkpoint_aug_epoch70.pth'
    path_to_result = 'results/result.jpg'

    # 0. Preprocess image
    remove_background(path_to_input_image, path_to_clean_image)

    # 1. Palm image rectification
    warp_result = warp(path_to_input_image, path_to_warped_image)
    if warp_result is None:
        print_error()
    else:
        remove_background(path_to_warped_image, path_to_warped_image_clean)
        resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value)

        # 2. Principal line detection
        net = UNet(n_channels=3, n_classes=1)
        net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        detect(net, path_to_warped_image_clean, path_to_palmline_image, resize_value)

        # 3. Line classification
        lines = classify(path_to_palmline_image)

        # 4. Length measurement
        im, contents = measure(path_to_warped_image_mini, lines)

        # 5. Save result
        save_result(im, contents, resize_value, path_to_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='the path to the input')
    args = parser.parse_args()
    main(args.input)




# import os
# import argparse
# import torch
# from tools import remove_background, resize, print_error, save_result
# from model import UNet
# from rectification import warp
# from detection import detect
# from classification import classify, measure
# from io import BytesIO
# from PIL import Image

# def main(input_image_path):
#     # Directories and Paths
#     results_dir = './results'
#     os.makedirs(results_dir, exist_ok=True)

#     resize_value = 256
#     paths = {
#         'clean_image': os.path.join(results_dir, 'palm_without_background.jpg'),
#         'warped_image': os.path.join(results_dir, 'warped_palm.jpg'),
#         'warped_image_clean': os.path.join(results_dir, 'warped_palm_clean.jpg'),
#         'warped_image_mini': os.path.join(results_dir, 'warped_palm_mini.jpg'),
#         'warped_image_clean_mini': os.path.join(results_dir, 'warped_palm_clean_mini.jpg'),
#         'palmline_image': os.path.join(results_dir, 'palm_lines.png'),
#         'result': os.path.join(results_dir, 'result.jpg')
#     }
#     model_path = 'checkpoint/checkpoint_aug_epoch70.pth'

#     # Load Model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = UNet(n_channels=3, n_classes=1).to(device)
#     net.load_state_dict(torch.load(model_path, map_location=device))
#     net.eval()

#     # Step 0: Remove background
#     remove_background(input_image_path, paths['clean_image'])

#     # Step 1: Palm image rectification
#     warp_result = warp(input_image_path, paths['warped_image'])
#     if warp_result is None:
#         print_error()
#         return

#     # Step 1.1: Clean background of the warped image and resize
#     remove_background(paths['warped_image'], paths['warped_image_clean'])
#     resize(paths['warped_image'], paths['warped_image_clean'], paths['warped_image_mini'], paths['warped_image_clean_mini'], resize_value)

#     # Step 2: Principal line detection
#     detect(net, paths['warped_image_clean'], paths['palmline_image'], resize_value, device)

#     # Step 3: Line classification
#     lines = classify(paths['palmline_image'])

#     # Step 4: Length measurement
#     im, contents = measure(paths['warped_image_mini'], lines)

#     # Step 5: Save the final result
#     save_result(im, contents, resize_value, paths['result'])

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', required=True, help='Path to the input image')
#     args = parser.parse_args()
#     main(args.input)
