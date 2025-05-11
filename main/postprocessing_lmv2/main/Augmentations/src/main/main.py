import cv2
import numpy as np
import os
from utility import draw_random_lines_across_image, masking, angle_mask, toggle_mask_all_corners
from PIL import Image
import os
from tqdm import tqdm
from constants import omitted_doc_type, random_agumentation, important_agumentation, max_random_agu, altra_max
import random
from augraphy import *
import cv2
import numpy as np
import random
from augraphy_agument import AugraphyAgument
import os


if __name__=="__main__":
    root_path = "/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/CI"
    for formats_ in os.listdir(root_path):
        formats_ = 'f14' #mention folder name
        for splits in os.listdir(os.path.join(root_path, formats_)):
            
            folder_path = os.path.join(root_path, formats_, splits)
            # for doc_type_ in range(0, 1):
            doc_type = splits #root_path.split('/')[-1]
            # folder_path = root_path #os.path.join(root_path, doc_type)
            img_lst= os.listdir(folder_path)
            img_lst= [img.split('.png')[0] for img in img_lst]
            print(img_lst)
            aug_folder= os.path.join(folder_path, f'{doc_type}_aug')
            os.makedirs(aug_folder, exist_ok=True)
            # exit('+++++++++++++++=')  
            for img in tqdm(img_lst, desc='Processing images', unit='image'):
                # Load an image from file
                random_number1 = random.randint(0, 4)
                image_path = os.path.join(folder_path,img+'.png')
                original_image = cv2.imread(image_path)

                augraphy_agument_obj = AugraphyAgument(original_image)
                print(img)
                # cv2.imwrite(os.path.join(aug_folder, img+"png"), original_image)
                cv2.imwrite(os.path.join(aug_folder, img + '.png'), original_image)
                
                lines_on_img = draw_random_lines_across_image(image_path, 16)
                
                masking_img = masking(image_path)
                


                mask_size = 150
                img_folds = toggle_mask_all_corners(image_path, mask_size)
                
                # Flip the image both horizontally and vertically
                flipped_both = cv2.flip(original_image, -1)

                # Display the original and flipped images
                # cv2.imshow('Original Image', original_image)
                # cv2.imshow('Flipped Vertical', flipped_vertical)
                # cv2.imshow('Flipped Horizontal', flipped_horizontal)
                # cv2.imshow('Flipped Both', flipped_both)
                # Scaling the image
                scale_factor = 1.5  # Adjust the scale factor as needed
                ####################################################################################
                ####################################################################################
                ####################################################################################
                ####################################################################################
                # scaled_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                # cv2.imwrite(os.path.join(aug_folder, img+"_scaled.png"), scaled_image)
                ###################################################################################
                ###################################################################################
                ###################################################################################
                # Get image dimensions
                height, width = original_image.shape[:2]

                # # Define the shearing parameters
                # shear_factor = 0.1  # Adjust the shear factor as needed

                # # Create an affine transformation matrix for shearing
                # shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)

                # # Apply the affine transformation to the image
                # sheared_image = cv2.warpAffine(original_image, shear_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

                # # Display the original and sheared images
                # # cv2.imshow('Original Image', original_image)
                # # cv2.imshow('Sheared Image', sheared_image)
                # cv2.imwrite(os.path.join(folder_path, img+"_sheared.png"), sheared_image)



                ################################################################
                ################################################################
                ################################################################
                ################################################################
                # Brightness and Contrast Adjustment
                # brightness_factor = 50  # Adjust the brightness factor as needed
                # contrast_factor = 1.5   # Adjust the contrast factor as needed

                # # Adjust brightness and contrast
                # adjusted_image = cv2.convertScaleAbs(original_image, alpha=contrast_factor, beta=brightness_factor)

                # # Display the adjusted image
                # # cv2.imshow('Adjusted Image', adjusted_image)
                # cv2.imwrite(os.path.join(aug_folder, img+"_adjust.png"), adjusted_image)
                ############################################################################
                ############################################################################
                ############################################################################

                # Salt-and-pepper noise parameters
                salt_and_pepper_ratio = 0.06  # Adjust the ratio as needed

                # Generate salt-and-pepper noise
                salt_and_pepper_mask = np.random.rand(*original_image.shape[:2])
                salt_pixels = salt_and_pepper_mask < salt_and_pepper_ratio / 2.0
                pepper_pixels = salt_and_pepper_mask > 1 - salt_and_pepper_ratio / 2.0

                # Add salt-and-pepper noise to the image
                
                noisy_image = original_image.copy()
                noisy_image[salt_pixels] = 255  # Set salt pixels to white (255)
                noisy_image[pepper_pixels] = 0  # Set pepper pixels to black (0)

                # Display the image with salt-and-pepper noise
                # cv2.imshow('Noisy Image with Salt-and-Pepper Noise', noisy_image)
                
                
                img_files = img.split(".png")[0]
                image = Image.open(image_path)
                
                # bbox value change 
                rotated_left = image.rotate(90, expand=True)  # expand=True maintains image size
                                
                rotated_right = image.rotate(-90, expand=True)

                # rotated_inverse = image.rotate(90, expand=True)
                # rotated_inverse = rotated_inverse.rotate(90, expand=True)

                backtowhite = Image.eval(image, lambda pixel: 255 - pixel)
        
                
                '''if doc_type not in omitted_doc_type:
                    rotated_left.save(os.path.join(aug_folder, f'{img_files}_l.png'))
                    rotated_right.save(os.path.join(aug_folder, f'{img_files}_r.png'))
                    angle_mask_img = angle_mask(image_path)
                    cv2.imwrite(os.path.join(aug_folder, img+"_angle.png"), angle_mask_img)
                    cv2.imwrite(os.path.join(aug_folder, img+"_folds.png"), img_folds)'''
                    
                    
                #important agumentations
                if doc_type in random_agumentation:
                    if not random_agumentation[doc_type]:
                        cv2.imwrite(os.path.join(aug_folder, img+"_lines.png"), lines_on_img)
                        cv2.imwrite(os.path.join(aug_folder, img+"_mask.png"), masking_img)
                        cv2.imwrite(os.path.join(aug_folder, img+"_flip.png"), flipped_both)
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        cv2.imwrite(os.path.join(aug_folder, img+"_noise.png"), noisy_image)
                        
                elif important_agumentation: #BOE
                    out_put, name = augraphy_agument_obj.watermark_()
                    cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    cv2.imwrite(os.path.join(aug_folder, img+"_lines.png"), lines_on_img)
                    cv2.imwrite(os.path.join(aug_folder, img+"_mask.png"), masking_img)
                    cv2.imwrite(os.path.join(aug_folder, img+"_noise.png"), noisy_image)
                    
                    cv2.imwrite(os.path.join(aug_folder, img+"_flip.png"), flipped_both)
                    
                    backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                    
                    out_put, name = augraphy_agument_obj.dirtydrum2_()
                    cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    
                    out_put, name = augraphy_agument_obj.dirty_rollers_()
                    cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    
                elif max_random_agu:
                    if random_number1 == 0:
                        out_put, name = augraphy_agument_obj.watermark_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        cv2.imwrite(os.path.join(aug_folder, img+"_lines.png"), lines_on_img)
                    elif random_number1 == 1:
                        cv2.imwrite(os.path.join(aug_folder, img+"_mask.png"), masking_img)
                        cv2.imwrite(os.path.join(aug_folder, img+"_noise.png"), noisy_image)
                        
                    elif random_number1 == 2:
                        cv2.imwrite(os.path.join(aug_folder, img+"_flip.png"), flipped_both)
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        
                        
                    elif random_number1 == 3:
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        out_put, name = augraphy_agument_obj.dirtydrum2_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    elif random_number1 == 4:
                        out_put, name = augraphy_agument_obj.dirtydrum2_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        
                        out_put, name = augraphy_agument_obj.dirty_rollers_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        
                elif altra_max: #AWB
                    if random_number1 == 0:
                        out_put, name = augraphy_agument_obj.watermark_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        cv2.imwrite(os.path.join(aug_folder, img+"_lines.png"), lines_on_img)
                    elif random_number1 == 1:
                        cv2.imwrite(os.path.join(aug_folder, img+"_mask.png"), masking_img)
                        cv2.imwrite(os.path.join(aug_folder, img+"_noise.png"), noisy_image)
                        
                        # ############## wrapup1 ##################
                        # augraphy_agument_obj = AugraphyAgument(original_image)
                        # out_put, name = augraphy_agument_obj.page_border1_()
                        # augraphy_agument_obj = AugraphyAgument(out_put)
                        # out_put, name = augraphy_agument_obj.BadPhotoCopy_type_5_()
                        # cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        # augraphy_agument_obj = AugraphyAgument(original_image)
                        
                        
                    elif random_number1 == 2:
                        cv2.imwrite(os.path.join(aug_folder, img+"_flip.png"), flipped_both)
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        out_put, name = augraphy_agument_obj.dirty_rollers_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        
                    elif random_number1 == 3:
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        out_put, name = augraphy_agument_obj.dirtydrum2_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                       
                        ############## wrapup2 ####################
                        augraphy_agument_obj = AugraphyAgument(original_image)
                        out_put, name = augraphy_agument_obj.binder_binding_clips_()
                        augraphy_agument_obj = AugraphyAgument(out_put)
                        out_put, name = augraphy_agument_obj.binder_binding_holes_()
                        augraphy_agument_obj = AugraphyAgument(out_put)
                        out_put, name = augraphy_agument_obj.binder_punch_holes_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        augraphy_agument_obj = AugraphyAgument(original_image)
                        
                    elif random_number1 == 4:
                        out_put, name = augraphy_agument_obj.dirtydrum2_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        out_put, name = augraphy_agument_obj.dirty_rollers_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                        
                    
                else:
                    if random_number1 == 0:
                        out_put, name = augraphy_agument_obj.watermark_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    elif random_number1 == 1:
                        cv2.imwrite(os.path.join(aug_folder, img+"_mask.png"), masking_img)
                        cv2.imwrite(os.path.join(aug_folder, img+"_noise.png"), noisy_image)
                        
                    elif random_number1 == 2:
                        cv2.imwrite(os.path.join(aug_folder, img+"_flip.png"), flipped_both)
                        
                    elif random_number1 == 3:
                        backtowhite.save(os.path.join(aug_folder, f'{img_files}_bw.png'))
                        out_put, name = augraphy_agument_obj.dirtydrum2_() 
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
                    elif random_number1 == 4:                    
                        out_put, name = augraphy_agument_obj.dirty_rollers_()
                        cv2.imwrite(os.path.join(aug_folder,f'{img_files}_{name}.png'), out_put)
    
        exit('>>>>>>>>>>>>>>>>>')

                        
                
                
                
                
                
                