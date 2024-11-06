from augraphy import *
import cv2
import numpy as np
import random
import os

class Extraction_AugraphyAgument():
    def __init__(self, image_):
        self.image = image_

    def watermark_(self):
            watermark= WaterMark(watermark_word = "random",
                        watermark_font_size = (10,15),
                        watermark_font_thickness = (20,25),
                        watermark_font_type = cv2.FONT_HERSHEY_SIMPLEX,
                        watermark_rotation = (0,360), 
                        watermark_location = "center", 
                        watermark_color = (0,0,255), 
                        watermark_method = "darken")
            return watermark(self.image), self.watermark_.__func__.__name__
            
    def BadPhotoCopy_new_op(self):
        BadPhotoCopy_type_new = BadPhotoCopy(noise_type=1,
                                   noise_side="none",
                                   noise_iteration=(2,3),
                                   noise_size=(2,2),
                                   noise_sparsity=(0.15,0.15),
                                   noise_concentration=(0.3,0.3),
                                   blur_noise=-1,
                                   blur_noise_kernel=(3, 3),
                                   wave_pattern=0,
                                   edge_effect=0)
        return BadPhotoCopy_type_new(self.image),self.BadPhotoCopy_new_op.__func__.__name__

    def binder_punch_holes_op(self):
        binder_punch_holes = BindingsAndFasteners(overlay_types="darken",
                                          foreground=None,
                                          effect_type="punch_holes",
                                          width_range = (70,80),
                                          height_range = (70,80),
                                          ntimes=(5, 5),
                                          nscales=(1.5, 1.5),
                                          edge="top",
                                          edge_offset=(30,50),
                                          use_figshare_library=0,
                                          )
        return binder_punch_holes(self.image),self.binder_punch_holes_op.__func__.__name__

    def bleedthrough_op(self):
        bleedthrough = BleedThrough(intensity_range=(0.5, 0.5),
                            color_range=(220, 224),
                            ksize=(5, 5),
                            sigmaX=0,
                            alpha=0.3,
                            offsets=(10, 20),
                        )
        return bleedthrough(self.image),self.bleedthrough_op.__func__.__name__

    def brightness_texturize_op(self):
        brightness_texturize = BrightnessTexturize(texturize_range=(0.9, 0.99),
                                           deviation=0.5 )
        return brightness_texturize(self.image),self.brightness_texturize_op.__func__.__name__

    def colorpaper_op(self):
        colorpaper= ColorPaper(hue_range=(0, 10), saturation_range=(10,30))
        return colorpaper(self.image),self.colorpaper_op.__func__.__name__

    def colorshift_op(self):
        colorshift = ColorShift(color_shift_offset_x_range = (3,5),
                        color_shift_offset_y_range = (3,5),
                        color_shift_iterations = (2,3),
                        color_shift_brightness_range = (0.9,1.1),
                        color_shift_gaussian_kernel_range = (3,3),
                        )


        return colorshift(self.image),self.colorshift_op.__func__.__name__

    def delaunay_pattern_op(self):
        delaunay_pattern = DelaunayTessellation(
                                                n_points_range = (500, 800),
                                                n_horizontal_points_range=(50, 100),
                                                n_vertical_points_range=(50, 100),
                                                noise_type = "random")
        return delaunay_pattern(self.image),self.delaunay_pattern_op.__func__.__name__

    def depthsimulatedblur_op(self):
        depthsimulatedblur = DepthSimulatedBlur(blur_center = "random",
                                        blur_major_axes_length_range = (400, 480),
                                        blur_minor_axes_length_range = (300, 500),
                                        )
        return depthsimulatedblur(self.image),self.depthsimulatedblur_op.__func__.__name__

    def dirtydrum_op(self):
        dirtydrum=DirtyDrum(line_width_range=(1, 4),
                      line_concentration=0.1,
                      direction=-1,
                      noise_intensity=0.3,
                      noise_value=(0, 30),
                      ksize=(3, 3),
                      sigmaX=0,
                      )
        return dirtydrum(self.image),self.dirtydrum_op.__func__.__name__

    def dirty_rollers_op(self):
        dirty_rollers=DirtyRollers(line_width_range=(12, 25),
                            scanline_type=0,
                            )
        return dirty_rollers(self.image),self.dirty_rollers_op.__func__.__name__

    def dotmatrix_op(self):
        dotmatrix=DotMatrix(dot_matrix_shape="circle",
                      dot_matrix_dot_width_range=(5, 5),
                      dot_matrix_dot_height_range=(5, 5),
                      dot_matrix_min_width_range=(1, 1),
                      dot_matrix_max_width_range=(50, 50),
                      dot_matrix_min_height_range=(1, 1),
                      dot_matrix_max_height_range=(50, 50),
                      dot_matrix_min_area_range=(10, 10),
                      dot_matrix_max_area_range=(800, 800),
                      dot_matrix_median_kernel_value_range = (29,29),
                      dot_matrix_gaussian_kernel_value_range=(1, 1),
                      dot_matrix_rotate_value_range=(0, 0)
                      )
        return dotmatrix(self.image),self.dotmatrix_op.__func__.__name__

    def hollow_op(self):
        hollow=Hollow(hollow_median_kernel_value_range = (101, 101),
                hollow_min_width_range=(1, 1),
                hollow_max_width_range=(200, 200),
                hollow_min_height_range=(1, 1),
                hollow_max_height_range=(200, 200),
                hollow_min_area_range=(10, 10),
                hollow_max_area_range=(5000, 5000),
                hollow_dilation_kernel_size_range = (3, 3),
                )
        return hollow(self.image),self.hollow_op.__func__.__name__



class shifting_annotation():
    def __init__(self,shifted_annotation_path,shifted_image_path,xshift,yshift) -> None:
        self.shifted_annotation_path = shifted_annotation_path
        self.shifted_image_path=shifted_image_path
        self.xshift=xshift
        self.yshift=yshift
        
    def load_annotations(self,file_path):
        with open(file_path, 'r') as file:
            annotations = file.readlines()
        return [line.strip().split() for line in annotations]        
        
    def save_annotations(self,annotations, file_path):
        with open(file_path, 'w+') as file:
            for ann in annotations:
                file.write(' '.join(map(str, ann)) + '\n')


    def shift_bounding_boxes(self,annotations, img_width, img_height, x_shift, y_shift):
        shifted_annotations = []
        for ann in annotations:
            class_id, x_center, y_center, width, height = map(float, ann)
            x_center = x_center * img_width + x_shift
            y_center = y_center * img_height + y_shift
            
            # Normalize the coordinates again
            x_center /= img_width
            y_center /= img_height
            
            shifted_annotations.append([int(class_id), x_center, y_center, width, height])
        return shifted_annotations

    def save_annotations(self,annotations, file_path):
        with open(file_path, 'w+') as file:
            for ann in annotations:
                file.write(' '.join(map(str, ann)) + '\n')


    def shift_objects_in_image(self,image, annotations, x_shift, y_shift):
        img_height, img_width = image.shape[:2]
        shifted_image = image.copy()
        for ann in annotations:
            class_id, x_center, y_center, width, height = map(float, ann)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            object_crop = image[y1:y2, x1:x2]
            cv2.imwrite("l.png",object_crop)
            
            new_x1 = max(0, x1 + x_shift)
            new_y1 = max(0, y1 + y_shift)
            new_x2 = new_x1 + object_crop.shape[1]
            new_y2 = new_y1 + object_crop.shape[0]
            # Ensure the new coordinates are within image bounds
            if new_x2 <= img_width and new_y2 <= img_height:
                shifted_image[y1:y2, x1:x2] = 255 
                shifted_image[new_y1:new_y2, new_x1:new_x2] = object_crop
                # Fill the old location with background color (optional)
                # assuming black background
        return shifted_image

    def shifting_op(self,image_file, annotation_file):
        filename = image_file.split("/")[-1]
        image = cv2.imread(image_file)
        annotations = self.load_annotations(annotation_file)
        img_height, img_width = image.shape[:2]
        shifted_annotations = self.shift_bounding_boxes(annotations, img_width, img_height, self.xshift, self.yshift)
        shifted_image = self.shift_objects_in_image(image, annotations, self.xshift, self.yshift)
        self.save_annotations(shifted_annotations, os.path.join(self.shifted_annotation_path,"Shifted_"+filename.replace(".png",".txt")))
        cv2.imwrite(os.path.join(self.shifted_image_path,"Shifted_"+filename),shifted_image)




