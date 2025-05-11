
import os
import cv2
from PIL import Image


def save_image(image, directory, filename):
    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Check if the file already exists, and if not, save the image
    file_path = os.path.join(directory, filename)
    image.save(file_path)




def image_rotation(root_path):
    filtered_images = []
    for img_name in os.listdir(root_path):
        print(img_name)
        # print(img_name.split('.')[0][-2:])
        if img_name.split('.')[0][-2:] not in ["_b", "_i", "_r", "_l"]:
            filtered_images.append(img_name)
    for i in filtered_images:
        image_path = os.path.join(root_path,i)
        img_files = i.split(".png")[0]
        image = Image.open(image_path)
        # bbox value change 
        rotated_left = image.rotate(90, expand=True)  # expand=True maintains image size
                        
        rotated_right = image.rotate(-90, expand=True)

        rotated_inverse = image.rotate(90, expand=True)
        rotated_inverse = rotated_inverse.rotate(90, expand=True)

        backtowhite = Image.eval(image, lambda pixel: 255 - pixel)
                    
        save_image(backtowhite, root_path, f'{img_files}_b.png')
        save_image(rotated_left, root_path, f'{img_files}_l.png')
        save_image(rotated_right, root_path, f'{img_files}_r.png')
        save_image(rotated_inverse, root_path, f'{img_files}_i.png')

        
        


def rotate90Deg(bndbox, img_width): # just passing width of image is enough for 90 degree rotation.
   x_min,y_min,x_max,y_max = bndbox
   new_xmin = y_min
   new_ymin = img_width-x_max
   new_xmax = y_max
   new_ymax = img_width-x_min
   return [new_xmin, new_ymin,new_xmax,new_ymax]


def rotate_90Deg( bndbox , image_width ):
    """
    image_width: Width of the image after clockwise rotation of 90 degrees
    """
    x_min,y_min,x_max,y_max = bndbox
    new_xmin = image_width - y_max # Reflection about center X-line
    new_ymin = x_min
    new_xmax = image_width - y_min # Reflection about center X-line
    new_ymax = x_max
    return [new_xmin, new_ymin,new_xmax,new_ymax]

def label_data_preparation(labels_path, imgs_path):
    filtered_labels = []
    for label_name in os.listdir(labels_path):
        print(label_name)
        # print(img_name.split('.')[0][-2:])
        if label_name.split('.')[0][-2:] not in ["_b", "_i", "_r", "_l"]:
            name_without_ext = label_name.split('.')[0]
            print(label_name)
            filtered_labels.append(label_name)
            # for j in ["_b", "_i", "_r", "_l"]:
            #     file_to_delete = os.path.join(labels_path, name_without_ext + j + '.txt')
            #     if os.path.exists(file_to_delete):
            #         os.remove(file_to_delete)
            #     print(file_to_delete)
    # label_files = os.listdir(labels_path)
    for i in filtered_labels:
        img_files = i.split(".txt")[0]
        print("img files: ", img_files)
        image_path = os.path.join(imgs_path, f'{img_files}.png')
        print(image_path)

        image = cv2.imread(image_path)
        h, w, _ = image.shape
        with open(os.path.join(labels_path, i), "r") as f:
            label = (f.read())
        label = label.split("\n")
        # label_path_delete = os.path.join(labels_path, i)
        # if os.path.exists(label_path_delete):
        #     os.remove(label_path_delete)
        append_flag = True
        for l in label:
            if append_flag:
                append_mode = "w"
            else:
                append_mode = "a" 
            l = l.split()
            if len(l) > 0:
                l_class = int(l[0])
                x_center = float(l[1]) * w
                y_center = float(l[2]) * h
                width = float(l[3]) * w
                height = int(float(l[4]) * h)
                x0 = int(x_center - (width/2))
                x1 = int(x_center + (width/2))
                y0 = int(y_center - (height / 2))
                y1 = int(y_center + (height / 2))    


                new_x1, new_y1, new_x2, new_y2 = rotate_90Deg([x0, y0, x1, y1], h) #right

                normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
                with open(os.path.join(labels_path,img_files+'_r'+'.txt'), append_mode) as file:
                    # Write the normalized values separated by spaces
                    file.write(" ".join(map(str, normalized_values)) + "\n")  


                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([x0, y0, x1, y1], w) #left

                normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
                with open(os.path.join(labels_path,img_files+'_l'+'.txt'), append_mode) as file:
                    # Write the normalized values separated by spaces
                    file.write(" ".join(map(str, normalized_values)) + "\n")              


                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([x0, y0, x1, y1], w) #inverse
                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([new_x1, new_y1, new_x2, new_y2], h)
                normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
                with open(os.path.join(labels_path,img_files+'_i'+'.txt'), append_mode) as file:
                    # Write the normalized values separated by spaces
                    file.write(" ".join(map(str, normalized_values)) + "\n")              




                normalized_values = [l_class, x0, y0, x1, y1] #black and white
                with open(os.path.join(labels_path,img_files+'_b'+'.txt'), append_mode) as file:
                    # Write the normalized values separated by spaces
                    file.write(" ".join(map(str, normalized_values)) + "\n")  



                normalized_values = [l_class, x0, y0, x1, y1] #original
                with open(os.path.join(labels_path,img_files+'.txt'), append_mode) as file:
                    # Write the normalized values separated by spaces
                    file.write(" ".join(map(str, normalized_values)) + "\n")
        
            append_flag = False
                    
            
            
            
                     
        