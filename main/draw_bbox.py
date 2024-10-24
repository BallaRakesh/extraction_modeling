
from PIL import Image, ImageDraw, ImageFont
import os
import cv2  # Assuming you are using OpenCV to handle images

image = Image.open("/home/ntlpt19/Desktop/TF_release/layoutlmv1/data/Images/82092117.png")
image = image.convert("RGB")
import json

with open('/home/ntlpt19/Desktop/TF_release/layoutlmv1/data/Labels/82092117.json') as f:
  data = json.load(f)

for annotation in data['form']:
  print(annotation)


draw = ImageDraw.Draw(image, "RGBA")

font = ImageFont.load_default()

label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

for annotation in data['form']:
  label = annotation['label']
  general_box = annotation['box']
  draw.rectangle(general_box, outline=label2color[label], width=2)
  draw.text((general_box[0] + 10, general_box[1] - 10), label, fill=label2color[label], font=font)
  words = annotation['words']
  for word in words:
    box = word['box']
    draw.rectangle(box, outline=label2color[label], width=1)



# Define the output directory and image name
output_dir = 'output_folder'
image_name = 'annotated_image.png'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the image with the drawings
image_path = os.path.join(output_dir, image_name)
image.save(image_path)