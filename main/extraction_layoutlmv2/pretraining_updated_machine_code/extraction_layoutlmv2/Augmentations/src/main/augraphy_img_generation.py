from augraphy import *
import cv2
import numpy as np
import random
from augraphy_agument import AugraphyAgument
import os


img_folder = '/home/ntlpt19/Downloads/Classification_final_training/testing'
out_folder = '/home/ntlpt19/Downloads/Classification_final_training/out_put'


img_lst= os.listdir(img_folder)
img_lst= [img.split('.png')[0] for img in img_lst]
for imgs in img_lst:
    image = cv2.imread(os.path.join(img_folder, imgs+'.png'))

    augraphy_agument_obj = AugraphyAgument(image)


    out_put, name = augraphy_agument_obj.BadPhotoCopy_type_5_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)
    
    out_put, name = augraphy_agument_obj.BadPhotoCopy_type_4_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.BadPhotoCopy_type_3_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.BadPhotoCopy_type_2_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)
    
    out_put, name = augraphy_agument_obj.BadPhotoCopy_type_1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.binder_binding_clips_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.user_binder_clips_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.binder_binding_holes_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.binder_punch_holes_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.bleedthrough_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.brightness_dimmer_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.colorpaper_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.colorshift_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.delaunay_pattern_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.depthsimulatedblur_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dirtydrum2_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dirtydrum3_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dirty_rollers_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dirtyscreen_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dirther_floyd_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.dotmatrix_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.doubleexposure_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.faxify_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.gamma_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)
    
    
    
    out_put, name = augraphy_agument_obj.hollow_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.inkbleed_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.inkcolorswap_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.inkmottling_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lcdscreenpattern_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.letterpress_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lighting_gradient_gaussian_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lighting_gradient_linear_static_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lines_degradation_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lines_degradation1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.low_ink_periodic_line_consistent_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.low_ink_periodic_line_non_consistent_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.lowlightnoise_obj_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.markup_strikethrough_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.markup_highlight_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.markup_underline_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.moire_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.noise_texturize_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.noisylines_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.quasi_pattern_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)
    
    
    out_put, name = augraphy_agument_obj.scribbles_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.shadowcast_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.subtle_noise_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.voronoi_pattern_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.watermark_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.book_binder_up_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.book_binder_down_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.folding_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.geometric_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.geometric1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.geometric2_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.geometric3_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.glitcheffect_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.glitcheffect1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.inkshifter_obj_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.page_border_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.page_border1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.rescale_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.rescale1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.rescale3_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)
    
    
    
    out_put, name = augraphy_agument_obj.sectionshift_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.sectionshift1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.squish_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)

    out_put, name = augraphy_agument_obj.squish1_()
    cv2.imwrite(os.path.join(out_folder, f'{imgs}_{name}.png'), out_put)