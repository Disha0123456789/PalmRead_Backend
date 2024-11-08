# import numpy as np
# from PIL import Image
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import cv2
# from pillow_heif import register_heif_opener

# def heic_to_jpeg(heic_dir, jpeg_dir):
#     register_heif_opener()
#     image = Image.open(heic_dir)
#     image.save(jpeg_dir, "JPEG")

# def remove_background(jpeg_dir, path_to_clean_image):
#     if jpeg_dir.lower().endswith('.heic'):
#         heic_to_jpeg(jpeg_dir, jpeg_dir[:-5] + '.jpg')
#         jpeg_dir = jpeg_dir[:-5] + '.jpg'
#     img = cv2.imread(jpeg_dir)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower = np.array([0, 20, 80], dtype="uint8")
#     upper = np.array([50, 255, 255], dtype="uint8")
#     mask = cv2.inRange(hsv, lower, upper)
#     result = cv2.bitwise_and(img, img, mask=mask)
#     b, g, r = cv2.split(result)
#     filter = g.copy()
#     ret, mask = cv2.threshold(filter, 10, 255, 1)
#     img[mask == 255] = 255
#     cv2.imwrite(path_to_clean_image, img)

# def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
#     pil_img = Image.open(path_to_warped_image)
#     pil_img_clean = Image.open(path_to_warped_image_clean)
#     pil_img.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_mini)
#     pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST).save(path_to_warped_image_clean_mini)

# def save_result(im, contents, resize_value, path_to_result):
#     if im is None:
#         print_error()
#     else:
#         heart_content_2, head_content_2, life_content_2, marriage_content_2, fate_content_2 = contents
#         image_height, image_width = im.size
#         fontsize = 12

#         plt.tick_params(
#             axis='both',
#             which='both',
#             bottom=False,
#             left=False,
#             labelbottom=False,
#             labelleft=False
#         )

#         note_1 = '* Note: This program is just for fun! Please take the result with a light heart.'
#         note_2 = '   If you want to check out more about palmistry, we recommend https://www.allure.com/story/palm-reading-guide-hand-lines'

#         plt.title('Check your palmistry result!', fontsize=14, y=1.01)

#         plt.text(image_width + 15, 15, "<Heart line>", color='r', fontsize=fontsize)
#         plt.text(image_width + 15, 55, heart_content_2, fontsize=fontsize)
#         plt.text(image_width + 15, 80, "<Head line>", color='g', fontsize=fontsize)
#         plt.text(image_width + 15, 120, head_content_2, fontsize=fontsize)
#         plt.text(image_width + 15, 145, "<Life line>", color='b', fontsize=fontsize)
#         plt.text(image_width + 15, 185, life_content_2, fontsize=fontsize)

#         plt.text(image_width + 15, 230, note_1, fontsize=fontsize-1, color='gray')
#         plt.text(image_width + 15, 250, note_2, fontsize=fontsize-1, color='gray')

#         fig, ax = plt.subplots()
#         ax.imshow(im)
#         plt.savefig(path_to_result, bbox_inches='tight')
#         plt.close(fig)

# def print_error():
#     print('Palm lines not properly detected! Please use another palm image.')


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pillow_heif import register_heif_opener

def heic_to_jpeg(heic_dir, jpeg_dir):
    register_heif_opener()
    image = Image.open(heic_dir)
    image.save(jpeg_dir, "JPEG")

def remove_background(jpeg_dir, path_to_clean_image):
    # Convert HEIC to JPEG if needed
    if jpeg_dir.lower().endswith('.heic'):
        heic_to_jpeg(jpeg_dir, jpeg_dir[:-5] + '.jpg')
        jpeg_dir = jpeg_dir[:-5] + '.jpg'

    # Load image and convert to HSV
    img = cv2.imread(jpeg_dir)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create mask based on HSV color range
    lower, upper = np.array([0, 20, 80], dtype="uint8"), np.array([50, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Further filtering to remove small artifacts
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    img[mask == 255] = 255  # Set background to white
    cv2.imwrite(path_to_clean_image, img)

def resize(path_to_warped_image, path_to_warped_image_clean, path_to_warped_image_mini, path_to_warped_image_clean_mini, resize_value):
    # Open images and resize simultaneously
    pil_img = Image.open(path_to_warped_image)
    pil_img_clean = Image.open(path_to_warped_image_clean)
    
    # Resize both images and save
    resized_img = pil_img.resize((resize_value, resize_value), resample=Image.NEAREST)
    resized_img_clean = pil_img_clean.resize((resize_value, resize_value), resample=Image.NEAREST)
    
    resized_img.save(path_to_warped_image_mini)
    resized_img_clean.save(path_to_warped_image_clean_mini)

def save_result(im, contents, resize_value, path_to_result):
    if im is None:
        print_error()
        return

    # Unpack contents
    heart_content_2, head_content_2, life_content_2, *_ = contents
    image_height, image_width = im.size
    
    # Set up Matplotlib parameters for faster plotting
    fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
    ax.imshow(im)
    ax.axis('off')  # Hide axes

    # Annotations
    annotations = [
        ("<Heart line>", 'r', heart_content_2, 15),
        ("<Head line>", 'g', head_content_2, 80),
        ("<Life line>", 'b', life_content_2, 145)
    ]
    for label, color, content, y_pos in annotations:
        plt.text(image_width + 15, y_pos, label, color=color, fontsize=12)
        plt.text(image_width + 15, y_pos + 40, content, fontsize=12)
    
    # Disclaimer text
    disclaimer_text = [
        ("* Note: This program is just for fun! Please take the result with a light heart.", 'gray', 230),
        ("If you want to check out more about palmistry, we recommend https://www.allure.com/story/palm-reading-guide-hand-lines", 'gray', 250)
    ]
    for note, color, y_pos in disclaimer_text:
        plt.text(image_width + 15, y_pos, note, fontsize=11, color=color)
    
    # Save result efficiently
    plt.savefig(path_to_result, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def print_error():
    print('Palm lines not properly detected! Please use another palm image.')
