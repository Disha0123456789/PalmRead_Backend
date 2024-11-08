# import os
# import json
# import random
# from PIL import Image, ImageDraw
# import cv2
# import mediapipe as mp

# def measure(path_to_warped_image_mini, lines):
#     heart_thres_x = 0
#     head_thres_x = 0
#     life_thres_y = 0

#     # Load content from JSON file
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     json_path = os.path.join(base_dir, 'palm_reading_content.json')
#     with open(json_path, 'r') as file:
#         palm_reading_content = json.load(file)

#     mp_hands = mp.solutions.hands
#     with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
#         image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
#         image_height, image_width, _ = image.shape

#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         hand_landmarks = results.multi_hand_landmarks[0]

#         zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
#         one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
#         five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
#         nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
#         thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

#         heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
#         head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
#         life_thres_y = image_height * (one + (zero - one) / 3)

#     im = Image.open(path_to_warped_image_mini)
#     width = 3
#     if (None in lines) or (len(lines) < 3):
#         return None, None
#     else:
#         draw = ImageDraw.Draw(im)

#         heart_line = lines[0]
#         head_line = lines[1]
#         life_line = lines[2]

#         heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
#         heart_line_tip = heart_line_points[0]
#         if heart_line_tip[0] < heart_thres_x:
#             heart_content_2 = random.choice(list(palm_reading_content['heart']['long'].values()))
#             marriage_content_2 = random.choice(list(palm_reading_content['marriage']['long'].values()))
#         else:
#             heart_content_2 = random.choice(list(palm_reading_content['heart']['short'].values()))
#             marriage_content_2 = random.choice(list(palm_reading_content['marriage']['short'].values()))
#         draw.line(heart_line_points, fill="red", width=width)

#         head_line_points = [tuple(reversed(l[:2])) for l in head_line]
#         head_line_tip = head_line_points[-1]
#         if head_line_tip[0] > head_thres_x:
#             head_content_2 = random.choice(list(palm_reading_content['head']['long'].values()))
#             fate_content_2 = random.choice(list(palm_reading_content['fate']['long'].values()))
#         else:
#             head_content_2 = random.choice(list(palm_reading_content['head']['short'].values()))
#             fate_content_2 = random.choice(list(palm_reading_content['fate']['short'].values()))
#         draw.line(head_line_points, fill="green", width=width)

#         life_line_points = [tuple(reversed(l[:2])) for l in life_line]
#         life_line_tip = life_line_points[-1]
#         if life_line_tip[1] > life_thres_y:
#             life_content_2 = random.choice(list(palm_reading_content['life']['long'].values()))
#         else:
#             life_content_2 = random.choice(list(palm_reading_content['life']['short'].values()))
#         draw.line(life_line_points, fill="blue", width=width)

#         contents = [heart_content_2, head_content_2, life_content_2, marriage_content_2, fate_content_2]
#         return im, contents

import os
import json
import random
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
import numpy as np

# Load JSON once to reduce redundant I/O
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, 'palm_reading_content.json')
with open(json_path, 'r') as file:
    palm_reading_content = json.load(file)

# Initialize Mediapipe hands only once
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def measure(path_to_warped_image_mini, lines):
    if not lines or len(lines) < 3 or None in lines:
        return None, None

    # Load image and prepare
    image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Process hand landmarks
    results = hands_detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    zero_y = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
    one_y = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
    five_x = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
    nine_x = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
    thirteen_x = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

    # Calculate thresholds
    heart_thres_x = image_width * (1 - (nine_x + (five_x - nine_x) * 2 / 5))
    head_thres_x = image_width * (1 - (thirteen_x + (nine_x - thirteen_x) / 3))
    life_thres_y = image_height * (one_y + (zero_y - one_y) / 3)

    # Prepare the output image
    im = Image.open(path_to_warped_image_mini)
    draw = ImageDraw.Draw(im)
    width = 3

    # Draw and classify lines
    contents = []

    for line, line_name, color, threshold, long_category, short_category in [
        (lines[0], 'heart', 'red', heart_thres_x, 'heart', 'marriage'),
        (lines[1], 'head', 'green', head_thres_x, 'head', 'fate'),
        (lines[2], 'life', 'blue', life_thres_y, 'life', None)
    ]:
        line_points = [tuple(reversed(l[:2])) for l in line]
        draw.line(line_points, fill=color, width=width)

        # Determine content based on line tips and threshold
        line_tip = line_points[0] if line_name == 'heart' else line_points[-1]
        threshold_check = line_tip[0] < threshold if line_name in ['heart', 'head'] else line_tip[1] > threshold

        # Fetch content from JSON
        content_key = 'long' if threshold_check else 'short'
        primary_content = random.choice(list(palm_reading_content[line_name][content_key].values()))
        contents.append(primary_content)

        # Additional content for heart line
        if line_name == 'heart' and short_category:
            secondary_content = random.choice(list(palm_reading_content[short_category][content_key].values()))
            contents.append(secondary_content)

    return im, contents
