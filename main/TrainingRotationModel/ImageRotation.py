import os 
import shutil
import cv2 

class RotateImages:
    def __init__(self):
        self.image_dataset_path = 'IMAGE DIRECTORY PATH'
        self.result_path = 'RESULT DIRECTORY PATH'
        os.makedirs(self.result_path,exist_ok=True)
        self.image_0_dataset_path = os.path.join(self.result_path,'img_0')
        os.makedirs(self.image_0_dataset_path,exist_ok=True) 
        self.image_90_dataset_path = os.path.join(self.result_path,'img_90')
        os.makedirs(self.image_90_dataset_path,exist_ok=True)
        self.image_180_dataset_path = os.path.join(self.result_path,'img_180')
        os.makedirs(self.image_180_dataset_path,exist_ok=True)
        self.image_270_dataset_path = os.path.join(self.result_path,'img_270')
        os.makedirs(self.image_270_dataset_path,exist_ok=True)
        
    def rotate_image_0(self):
        for file in os.listdir(self.image_dataset_path):
            image_name = file.split(".")[0]
            image_name = image_name + 'rot0.png'
            file_path = os.path.join(self.image_dataset_path,file)
            if os.path.exists(file_path):
                result_path = os.path.join(self.image_0_dataset_path,image_name)
                shutil.copy(file_path,result_path)
    def rotate_image_90(self):
        for file in os.listdir(self.image_dataset_path):
            image_name = file.split(".")[0]
            image_name = image_name + '_rot90.png'
            file_path = os.path.join(self.image_dataset_path,file)
            if os.path.exists(file_path):
                result_path = os.path.join(self.image_90_dataset_path,image_name)
                image = cv2.imread(file_path)
                if image is not None:
                    result_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    if result_image is not None:
                        cv2.imwrite(result_path,result_image)            
    def rotate_image_180(self):
        for file in os.listdir(self.image_dataset_path):
            image_name = file.split(".")[0]
            image_name = image_name + '_rot180.png'
            file_path = os.path.join(self.image_dataset_path,file)
            if os.path.exists(file_path):
                result_path = os.path.join(self.image_180_dataset_path,image_name)
                image = cv2.imread(file_path)
                if image is not None:
                    result_image = cv2.rotate(image, cv2.ROTATE_180)
                    if result_image is not None:
                        cv2.imwrite(result_path,result_image)     
                
    def rotate_image_270(self):
        for file in os.listdir(self.image_dataset_path):
            image_name = file.split(".")[0]
            image_name = image_name + '_rot270.png'
            file_path = os.path.join(self.image_dataset_path,file)
            if os.path.exists(file_path):
                result_path = os.path.join(self.image_270_dataset_path,image_name)
                image = cv2.imread(file_path)
                if image is not None:
                    result_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    if result_image is not None:
                        cv2.imwrite(result_path,result_image)  
                        
if __name__ == '__main__':
    rotation_object = RotateImages()
    rotation_object.rotate_image_0()
    rotation_object.rotate_image_90()
    rotation_object.rotate_image_180()
    rotation_object.rotate_image_270()
    print("==============================================")
    print("IMAGE ROTATION HAS BEEN SUCCESSFULLY COMPLETED")
    print("==============================================")