from PIL import Image
import os


if __name__=="__main__": 
    root_path: str = '/home/ntlpt19/Downloads/Evaluation_Data/testing_aug/AWB'
    images_fol: list = os.listdir(root_path)

    for i in images_fol:
        image_path = os.path.join(root_path,i)
        img_files = i.split(".png")[0]
        image = Image.open(image_path)
        
        # bbox value change 
        rotated_left = image.rotate(90, expand=True)  # expand=True maintains image size
                        
        rotated_right = image.rotate(-90, expand=True)

        rotated_inverse = image.rotate(90, expand=True)
        rotated_inverse = rotated_inverse.rotate(90, expand=True)

        backtowhite = Image.eval(image, lambda pixel: 255 - pixel)
   
        backtowhite.save(os.path.join(root_path, f'{img_files}_bw.png'))
        rotated_left.save(os.path.join(root_path, f'{img_files}_l.png'))
        rotated_right.save(os.path.join(root_path, f'{img_files}_r.png'))
        rotated_inverse.save(os.path.join(root_path, f'{img_files}_i.png'))
        