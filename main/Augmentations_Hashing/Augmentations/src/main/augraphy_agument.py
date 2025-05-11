from augraphy import *
import cv2
import numpy as np
import random

class AugraphyAgument():
    def __init__(self, image_):
        self.image = image_
    
    def BadPhotoCopy_type_5_(self):
        BadPhotoCopy_type_5 = BadPhotoCopy(noise_type=5,
                                                    noise_side="all",
                                                    noise_iteration=(1,1),
                                                    noise_size=(1,3),
                                                    noise_value=(32, 128),
                                                    noise_sparsity=(0.3,0.3),
                                                    noise_concentration=(0.5,0.5),
                                                    blur_noise=0,
                                                    wave_pattern=0,
                                                    edge_effect=0)

        return BadPhotoCopy_type_5(self.image), self.BadPhotoCopy_type_5_.__func__.__name__

    def BadPhotoCopy_type_4_(self):
        BadPhotoCopy_type_4 = BadPhotoCopy(noise_type=4,
                                   noise_side="none",
                                   noise_iteration=(1,1),
                                   noise_size=(1,3),
                                   noise_value=(32, 255),
                                   noise_sparsity=(0.5,0.5),
                                   noise_concentration=(0.99,0.99),
                                   blur_noise=0,
                                   wave_pattern=0,
                                   edge_effect=0)
        return BadPhotoCopy_type_4(self.image), self.BadPhotoCopy_type_4_.__func__.__name__
        
        
    def BadPhotoCopy_type_3_(self):
        BadPhotoCopy_type_3 = BadPhotoCopy(noise_type=3,
                                   noise_side="none",
                                   noise_iteration=(1,1),
                                   noise_size=(1,3),
                                   noise_value=(128, 255),
                                   noise_sparsity=(0.2,0.3),
                                   noise_concentration=(0.5,0.5),
                                   blur_noise=1,
                                   blur_noise_kernel=(5, 5),
                                   wave_pattern=0,
                                   edge_effect=1)
        return BadPhotoCopy_type_3(self.image), self.BadPhotoCopy_type_3_.__func__.__name__
        
        
        
    def BadPhotoCopy_type_2_(self):
        BadPhotoCopy_type_2 = BadPhotoCopy(noise_type=2,
                                   noise_side="right",
                                   noise_iteration=(1,1),
                                   noise_size=(1,1),
                                   noise_sparsity=(0.4,0.5),
                                   noise_concentration=(0.2,0.2),
                                   blur_noise=1,
                                   blur_noise_kernel=(5, 5),
                                   wave_pattern=0,
                                   edge_effect=1)
        return BadPhotoCopy_type_2(self.image), self.BadPhotoCopy_type_2_.__func__.__name__
        
        
        
    def BadPhotoCopy_type_1_(self):
        BadPhotoCopy_type_1 = BadPhotoCopy(noise_type=1,
                                   noise_side="left",
                                   noise_iteration=(2,3),
                                   noise_size=(2,3),
                                   noise_sparsity=(0.15,0.15),
                                   noise_concentration=(0.3,0.3),
                                   blur_noise=-1,
                                   blur_noise_kernel=(5, 5),
                                   wave_pattern=0,
                                   edge_effect=0)
        
        return BadPhotoCopy_type_1(self.image), self.BadPhotoCopy_type_1_.__func__.__name__
        
        
    def binder_binding_clips_(self):
        binder_binding_clips = BindingsAndFasteners(overlay_types="darken",
                                            foreground=None,
                                            effect_type="clips",
                                            width_range = "random",
                                            height_range = "random",
                                            ntimes= (2, 3),
                                            nscales=(1, 2),
                                            edge="random",
                                            edge_offset=(10,20),
                                            use_figshare_library=0,
                                            )
        
        return binder_binding_clips(self.image), self.binder_binding_clips_.__func__.__name__
        
        
    def user_binder_clips_(self):
        binder_rectangle = np.full((50,20),fill_value=250,dtype="uint8")
        binder_rectangle[10:40,5:15] = 0


        user_binder_clips = BindingsAndFasteners(overlay_types="darken",
                                                    foreground=binder_rectangle,
                                                    ntimes= (2, 3),
                                                    nscales=(1, 2),
                                                    edge="right",
                                                    edge_offset=(10,20),
                                                    use_figshare_library=0,
                                                    )
        return user_binder_clips(self.image), self.user_binder_clips_.__func__.__name__
        
        
    def binder_binding_holes_(self):
        binder_binding_holes = BindingsAndFasteners(overlay_types="darken",
                                            foreground=None,
                                            effect_type="binding_holes",
                                            width_range = "random",
                                            height_range = "random",
                                            ntimes=(9, 10),
                                            nscales=(1, 2),
                                            edge="top",
                                            edge_offset=(40,50),
                                            use_figshare_library=0,
                                            )
        
        return binder_binding_holes(self.image), self.binder_binding_holes_.__func__.__name__
        
        
    def binder_punch_holes_(self):
        binder_punch_holes = BindingsAndFasteners(overlay_types="darken",
                                          foreground=None,
                                          effect_type="punch_holes",
                                          width_range = (70,80),
                                          height_range = (70,80),
                                          ntimes=(3, 3),
                                          nscales=(1.5, 1.5),
                                          edge="left",
                                          edge_offset=(30,50),
                                          use_figshare_library=0,
                                          )
        return binder_punch_holes(self.image), self.binder_punch_holes_.__func__.__name__
        
        
        
    def bleedthrough_(self):
        bleedthrough = BleedThrough(intensity_range=(0.1, 0.2),
                            color_range=(0, 224),
                            ksize=(17, 17),
                            sigmaX=0,
                            alpha=0.3,
                            offsets=(10, 20),
                        )
        
        return bleedthrough(self.image), self.bleedthrough_.__func__.__name__
        
        
    def brightness_dimmer_(self):
        brightness_dimmer= Brightness(brightness_range=(0.2, 0.8),
                                min_brightness=1,
                                min_brightness_value=(120, 150),
                        )
        return brightness_dimmer(self.image), self.brightness_dimmer_.__func__.__name__
        
        
    def brightness_brighten_(self):
        brightness_brighten= Brightness(brightness_range=(1.5, 2),
                       min_brightness=0,
                    )
        return brightness_brighten(self.image), self.brightness_brighten_.__func__.__name__
        
        
    def colorpaper_(self):
        colorpaper= ColorPaper(hue_range=(0, 10), saturation_range=(10,30))
        
        return colorpaper(self.image), self.colorpaper_.__func__.__name__
        
        
    def colorshift_(self):
        colorshift = ColorShift(color_shift_offset_x_range = (3,5),
                        color_shift_offset_y_range = (3,5),
                        color_shift_iterations = (2,3),
                        color_shift_brightness_range = (0.9,1.1),
                        color_shift_gaussian_kernel_range = (3,3),
                        )
        return colorshift(self.image), self.colorshift_.__func__.__name__
        
        
        
    def delaunay_pattern_(self):
        delaunay_pattern = DelaunayTessellation(
                                        n_points_range = (500, 800),
                                        n_horizontal_points_range=(50, 100),
                                        n_vertical_points_range=(50, 100),
                                        noise_type = "random")
        
        return delaunay_pattern(self.image), self.delaunay_pattern_.__func__.__name__
        
        
    def depthsimulatedblur_(self):
        depthsimulatedblur = DepthSimulatedBlur(blur_center = "random",
                                        blur_major_axes_length_range = (120, 200),
                                        blur_minor_axes_length_range = (120, 200),
                                        )
        return depthsimulatedblur(self.image), self.depthsimulatedblur_.__func__.__name__
        
        
        
    def dirtydrum2_(self):
        dirtydrum2 = DirtyDrum(line_width_range=(5, 10),
                      line_concentration=0.3,
                      direction=1,
                      noise_intensity=0.2,
                      noise_value=(0, 10),
                      ksize=(3, 3),
                      sigmaX=0,
                      )
        return dirtydrum2(self.image), self.dirtydrum2_.__func__.__name__
        
        
        
    def dirtydrum3_(self):
        dirtydrum3 = DirtyDrum(line_width_range=(2, 5),
                      line_concentration=0.3,
                      direction=2,
                      noise_intensity=0.4,
                      noise_value=(0, 5),
                      ksize=(3, 3),
                      sigmaX=0,
                      )
        return dirtydrum3(self.image), self.dirtydrum3_.__func__.__name__
        
        
        
    def dirty_rollers_(self):
        dirty_rollers = DirtyRollers(line_width_range=(12, 25),
                            scanline_type=0,
                            )
        return dirty_rollers(self.image), self.dirty_rollers_.__func__.__name__
        
        
        
    def dirtyscreen_(self):
        dirtyscreen = DirtyScreen(n_clusters = (50,100),
                                    n_samples = (2,20),
                                    std_range = (1,5),
                                    value_range = (150,250),
                                        )
        return dirtyscreen(self.image), self.dirtyscreen_.__func__.__name__
        
        
    # def dirther_ordered_(self):
    #     dirther_ordered = Dithering(dither="ordered",
    #                         order=8,
    #                         )
    def dirther_floyd_(self):
        dirther_floyd = Dithering(dither="floyd")
        return dirther_floyd(self.image), self.dirther_floyd_.__func__.__name__
        
        
        
        
    def dotmatrix_(self):
        dotmatrix = DotMatrix(dot_matrix_shape="circle",
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
        return dotmatrix(self.image), self.dotmatrix_.__func__.__name__
        
        
        
    def doubleexposure_(self):
        doubleexposure = DoubleExposure(gaussian_kernel_range=(9,12),
                                offset_direction=1,
                                offset_range=(18,25),
                                )
        return doubleexposure(self.image), self.doubleexposure_.__func__.__name__
        
        
        
    def faxify_(self):
        faxify = Faxify(scale_range = (1,2),
                monochrome = 1,
                monochrome_method = "cv2.threshold",
                monochrome_arguments = {"thresh":128, "maxval":128, "type":cv2.THRESH_BINARY},
                halftone = 1,
                invert = 1,
                half_kernel_size = (2,2),
                angle = (0, 360),
                sigma = (1,3))
        return faxify(self.image), self.faxify_.__func__.__name__
        
        
        
        
    def gamma_(self):
        gamma = Gamma(gamma_range=(2.0, 3.0))
        
        return gamma(self.image), self.gamma_.__func__.__name__
        
        
        
    def hollow_(self):
        hollow = Hollow(hollow_median_kernel_value_range = (101, 101),
                hollow_min_width_range=(1, 1),
                hollow_max_width_range=(200, 200),
                hollow_min_height_range=(1, 1),
                hollow_max_height_range=(200, 200),
                hollow_min_area_range=(10, 10),
                hollow_max_area_range=(5000, 5000),
                hollow_dilation_kernel_size_range = (3, 3),
                )
        return hollow(self.image), self.hollow_.__func__.__name__
        
        
        
    def inkbleed_(self):
        inkbleed = InkBleed(intensity_range=(0.4, 0.7),
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
                    )
        return inkbleed(self.image), self.inkbleed_.__func__.__name__
        
        
        
        
    def inkcolorswap_(self):
        inkcolorswap= InkColorSwap(ink_swap_color = "random",
                           ink_swap_sequence_number_range = (1,10),
                           ink_swap_min_width_range=(3,3),
                           ink_swap_max_width_range=(100,100),
                           ink_swap_min_height_range=(3,3),
                           ink_swap_max_height_range=(100,100),
                           ink_swap_min_area_range=(10,10),
                           ink_swap_max_area_range=(400,400)
                           )
        return inkcolorswap(self.image), self.inkcolorswap_.__func__.__name__
        
        
        
    def inkmottling_(self):
        inkmottling= InkMottling(ink_mottling_alpha_range=(0.5, 0.5),
                         ink_mottling_noise_scale_range=(1,1),
                         ink_mottling_gaussian_kernel_range=(3,5),
                         )
        
        return inkmottling(self.image), self.inkmottling_.__func__.__name__
        
    
    def lcdscreenpattern_(self):
        lcdscreenpattern = LCDScreenPattern(pattern_type="Horizontal",
                                    pattern_value_range = (0,16),
                                    pattern_skip_distance_range = (3,5),
                                    pattern_overlay_method = "darken",
                                    )
        
        return lcdscreenpattern(self.image), self.lcdscreenpattern_.__func__.__name__
        
        
    def letterpress_(self):
        letterpress = Letterpress(n_samples=(200, 500),
                          n_clusters=(300, 800),
                          std_range=(1500, 5000),
                          value_range=(200, 255),
                          value_threshold_range=(128, 128),
                          blur=1
                          )
        
        return letterpress(self.image), self.letterpress_.__func__.__name__
        
        
    def lighting_gradient_gaussian_(self):
        lighting_gradient_gaussian = LightingGradient(light_position=None,
                                              direction=90,
                                              max_brightness=255,
                                              min_brightness=0,
                                              mode="gaussian",
                                              transparency=0.5
                                              )
        
        return lighting_gradient_gaussian(self.image), self.lighting_gradient_gaussian_.__func__.__name__
        
        
        
        
    def lighting_gradient_linear_static_(self):
        lighting_gradient_linear_static = LightingGradient(light_position=None,
                                              direction=45,
                                              max_brightness=255,
                                              min_brightness=0,
                                              mode="linear_static",
                                              linear_decay_rate = 0.5,
                                              transparency=0.5
                                              )
        return lighting_gradient_linear_static(self.image), self.lighting_gradient_linear_static_.__func__.__name__
        
        
        
        
    def lines_degradation_(self):
        lines_degradation = LinesDegradation(line_roi = (0.0, 0.0, 1.0, 1.0),
                                     line_gradient_range=(32, 255),
                                     line_gradient_direction= (1,1),
                                     line_split_probability=(0.2, 0.3),
                                     line_replacement_value=(250, 250),
                                     line_min_length=(15, 15),
                                     line_long_to_short_ratio = (3,3),
                                     line_replacement_probability = (0.5, 0.5),
                                     line_replacement_thickness = (1, 2)
                                     )
        
        return lines_degradation(self.image), self.lines_degradation_.__func__.__name__
        
        
        
    def lines_degradation1_(self):
        lines_degradation1 = LinesDegradation(line_roi = (0.0, 0.0, 0.5, 1.0),
                                     line_gradient_range=(32, 255),
                                     line_gradient_direction= (2,2),
                                     line_split_probability=(0.2, 0.3),
                                     line_replacement_value=(0, 25),
                                     line_min_length=(15, 15),
                                     line_long_to_short_ratio = (3,3),
                                     line_replacement_probability = (1.0, 1.0),
                                     line_replacement_thickness = (2, 2)
                                     )
        return lines_degradation1(self.image), self.lines_degradation1_.__func__.__name__
        
        
        
        
    def low_ink_periodic_line_consistent_(self):
        low_ink_periodic_line_consistent =  LowInkPeriodicLines(count_range=(2, 5),
                                                        period_range=(30, 30),
                                                        use_consistent_lines=True,
                                                        noise_probability=0.1,
                                                        )
        
        return low_ink_periodic_line_consistent(self.image), self.low_ink_periodic_line_consistent_.__func__.__name__
        
        
        
    def low_ink_periodic_line_non_consistent_(self):
        low_ink_periodic_line_non_consistent =  LowInkPeriodicLines(count_range=(2, 5),
                                                        period_range=(10, 30),
                                                        use_consistent_lines=False,
                                                        noise_probability=0.1,
                                                        )
        return low_ink_periodic_line_non_consistent(self.image), self.low_ink_periodic_line_non_consistent_.__func__.__name__
        
        
        
    def lowlightnoise_obj_(self):
        lowlightnoise_obj = LowLightNoise(
                                num_photons_range = (50, 100),
                                alpha_range = (0.7, 0.9),
                                beta_range = (10, 30),
                                gamma_range = (1.0 , 1.8)
                            )
        
        return lowlightnoise_obj(self.image), self.lowlightnoise_obj_.__func__.__name__
        
        
    def markup_strikethrough_(self):
        markup_strikethrough = Markup(num_lines_range=(5, 7),
                              markup_length_range=(0.5, 1),
                              markup_thickness_range=(1, 2),
                              markup_type="strikethrough",
                              markup_ink = "pencil",
                              markup_color=(0, 0, 255),
                              repetitions=4,
                              large_word_mode=1,
                              single_word_mode=False)
        return markup_strikethrough(self.image), self.markup_strikethrough_.__func__.__name__
        
        
        
    def markup_highlight_(self):
        markup_highlight = Markup(num_lines_range=(1, 1),
                          markup_length_range=(0.5, 1),
                          markup_thickness_range=(5, 5),
                          markup_type="highlight",
                          markup_ink="highlighter",
                          markup_color=(0, 255, 0),
                          repetitions=1,
                          large_word_mode=1,
                          single_word_mode=False)
        
        return markup_highlight(self.image), self.markup_highlight_.__func__.__name__
        
        
        
    def markup_underline_(self):
        markup_underline = Markup(num_lines_range=(1, 1),
                          markup_length_range=(0.5, 1),
                          markup_thickness_range=(2, 2),
                          markup_type="underline",
                          markup_ink="marker",
                          markup_color=(255, 0, 0),
                          repetitions=1,
                          large_word_mode=1,
                          single_word_mode=False)
        
        return markup_underline(self.image), self.markup_underline_.__func__.__name__
        
        
        
    def moire_(self):
        moire = Moire(moire_density = (15,20),
              moire_blend_method = "normal",
              moire_blend_alpha = 0.1,
             )
        
        return moire(self.image), self.moire_.__func__.__name__
        
        
    def noise_texturize_(self):
        noise_texturize = NoiseTexturize(sigma_range=(2, 3),
                                 turbulence_range=(2, 5),
                                 texture_width_range=(300, 500),
                                 texture_height_range=(50, 500),
                                 )
        return noise_texturize(self.image), self.noise_texturize_.__func__.__name__
        
    def noisylines_(self):
        noisylines = NoisyLines(noisy_lines_direction = 0,
                        noisy_lines_location = "random",
                        noisy_lines_number_range = (3,5),
                        noisy_lines_color = (0,0,0),
                        noisy_lines_thickness_range = (2,2),
                        noisy_lines_random_noise_intensity_range = (0.01, 0.1),
                        noisy_lines_length_interval_range = (0,100),
                        noisy_lines_gaussian_kernel_value_range = (3,3),
                        noisy_lines_overlay_method = "ink_to_paper",
                        )
        return noisylines(self.image), self.noisylines_.__func__.__name__
        
    def quasi_pattern_(self):
        quasi_pattern = PatternGenerator(
                                imgx = 512,
                                imgy= 512,
                                n_rotation_range = (10,15)
                            )
        return quasi_pattern(self.image), self.quasi_pattern_.__func__.__name__
        
        
    # def reflected_light_(self):
        #     reflected_light = NoiseTexturize(reflected_light_smoothness = 0.8,
        #                              reflected_light_internal_radius_range=(0.0, 0.2),
        #                              reflected_light_external_radius_range=(0.1, 0.8),
        #                              reflected_light_minor_major_ratio_range = (0.9, 1.0),
        #                              reflected_light_color = (255,255,255),
        #                              reflected_light_internal_max_brightness_range=(0.9,1.0),
        #                              reflected_light_external_max_brightness_range=(0.9,0.9),
        #                              reflected_light_location = "random",
        #                              reflected_light_ellipse_angle_range = (0, 360),
        #                              reflected_light_gaussian_kernel_size_range = (5,310),
        #                              )
        
    def scribbles_(self):
        scribbles = Scribbles(scribbles_type="random",
                      scribbles_ink="random",
                      scribbles_location="random",
                      scribbles_size_range=(400, 600),
                      scribbles_count_range=(1, 6),
                      scribbles_thickness_range=(1, 3),
                      scribbles_brightness_change=[8, 16],
                      scribbles_skeletonize=0,
                      scribbles_skeletonize_iterations=(2, 3),
                      scribbles_color="random",
                      scribbles_text="random",
                      scribbles_text_font="random",
                      scribbles_text_rotate_range=(0, 360),
                      scribbles_lines_stroke_count_range=(1, 6),
                      )
        
        return scribbles(self.image), self.scribbles_.__func__.__name__
        
        
    def shadowcast_(self):
        shadowcast = ShadowCast(shadow_side = "bottom",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.5,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (101, 301),
                        )
        
        return shadowcast(self.image), self.shadowcast_.__func__.__name__
        
        
        
    def subtle_noise_(self):
        subtle_noise = SubtleNoise(subtle_range=25)
        
        return subtle_noise(self.image), self.subtle_noise_.__func__.__name__
        
        
        
    def voronoi_pattern_(self):
        voronoi_pattern = VoronoiTessellation(
                         mult_range = (50,80),
                         seed = 19829813472 ,
                         num_cells_range = (500,800),
                         noise_type = "random",
                         background_value = (200, 256)
                        )
        return voronoi_pattern(self.image), self.voronoi_pattern_.__func__.__name__
        
        
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
        
        
    def book_binder_up_(self):
        book_binder_up = BookBinding(shadow_radius_range=(100, 100),
                             curve_range_right=(300, 300),
                             curve_range_left=(300, 300),
                             curve_ratio_right = (0.3, 0.3),
                             curve_ratio_left = (0.3, 0.3),
                             mirror_range=(1.0, 1.0),
                             binding_align = 0,
                             binding_pages = (10,10),
                             curling_direction=0,
                             backdrop_color=(255, 255, 255),
                             enable_shadow=1,
                             use_cache_images = 0,
                             )
        return book_binder_up(self.image), self.book_binder_up_.__func__.__name__
        
        
    def book_binder_down_(self):
        book_binder_down = BookBinding(shadow_radius_range=(100, 100),
                              curve_range_right=(50, 50),
                              curve_range_left=(300, 300),
                              curve_ratio_right = (0.05, 0.05),
                              curve_ratio_left = (0.3, 0.3),
                              mirror_range=(0.50, 0.50),
                              binding_align = 1,
                              binding_pages = (10,10),
                              curling_direction=1,
                              backdrop_color=(255, 255, 255),
                              enable_shadow=0,
                              use_cache_images = 0,
                              )
        return book_binder_down(self.image), self.book_binder_down_.__func__.__name__
        
        
        
    def folding_(self):
        folding = Folding(fold_count=10,
                  fold_noise=0.0,
                  fold_angle_range = (-360,360),
                  gradient_width=(0.1, 0.2),
                  gradient_height=(0.01, 0.1),
                  backdrop_color = (0,0,0),
                  )
        return folding(self.image), self.folding_.__func__.__name__
        
        
    def geometric_(self):
        geometric = Geometric(scale=(0.5, 1.5),
                      translation=(50, -50),
                      fliplr=1,
                      flipud=1,
                      crop=(),
                      rotate_range=(3, 5)
                      )
        return geometric(self.image), self.geometric_.__func__.__name__
        
    def geometric1_(self):
        geometric1 = Geometric(rotate_range=(-10,10))
        
        return geometric1(self.image), self.geometric1_.__func__.__name__
        
        
    def geometric2_(self):
        geometric2 = Geometric(crop=(0.2, 0.2, 0.8, 0.8))
        
        return geometric2(self.image), self.geometric2_.__func__.__name__
        
        
    def geometric3_(self):
        geometric3 = Geometric(fliplr=1, flipud=1)
        
        return geometric3(self.image), self.geometric3_.__func__.__name__
        
        
    def glitcheffect_(self):
        glitcheffect= GlitchEffect(glitch_direction = "horizontal",
                           glitch_number_range = (8, 16),
                           glitch_size_range = (5, 50),
                           glitch_offset_range = (5, 10)
                           )
        return glitcheffect(self.image), self.glitcheffect_.__func__.__name__
        
        
    def glitcheffect1_(self):
        glitcheffect1 = GlitchEffect()
        return glitcheffect1(self.image), self.glitcheffect1_.__func__.__name__
        
        
        
    def inkshifter_obj_(self):
        inkshifter_obj = InkShifter(
                            text_shift_scale_range=(18, 27),
                            text_shift_factor_range=(1, 4),
                            text_fade_range=(0, 2),
                            noise_type = "random",
                        )
        
        return inkshifter_obj(self.image), self.inkshifter_obj_.__func__.__name__
        
        
        
    def page_border_(self):
        page_border = PageBorder(page_border_width_height = (30, -40),
                         page_border_color=(0, 0, 0),
                         page_border_background_color=(255, 255, 255),
                         page_border_use_cache_images = 0,
                         page_border_trim_sides = (0, 0, 0, 0),
                         page_numbers = 10,
                         page_rotate_angle_in_order = 0,
                         page_rotation_angle_range = (1, 5),
                         curve_frequency=(0, 1),
                         curve_height=(1, 2),
                         curve_length_one_side=(30, 60),
                         same_page_border=0,
                         )
        
        return page_border(self.image), self.page_border_.__func__.__name__
        
        
    def page_border1_(self):
        page_border1 = PageBorder()
        return page_border1(self.image), self.page_border1_.__func__.__name__
        
    def rescale_(self):
        rescale = Rescale(target_dpi=300)
        
        pipeline1 = AugraphyPipeline(pre_phase=[rescale], ink_phase=[InkBleed()], paper_phase=[ColorPaper()], post_phase=[BleedThrough()], fixed_dpi=1)
        return pipeline1(self.image), self.rescale_.__func__.__name__
        
    def rescale1_(self):
        rescale = Rescale(target_dpi=300)
        pipeline2 = AugraphyPipeline(pre_phase=[rescale], ink_phase=[InkBleed()], paper_phase=[ColorPaper()], post_phase=[BleedThrough()], fixed_dpi=0)
        return pipeline2(self.image), self.rescale1_.__func__.__name__
        
    def rescale3_(self):
        rescale3 = Rescale(target_dpi=300)
        return rescale3(self.image), self.rescale3_.__func__.__name__
        
        
    def sectionshift_(self):
        sectionshift = SectionShift(section_shift_number_range = (5,5),
                            section_shift_locations = "random",
                            section_shift_x_range = (20,20),
                            section_shift_y_range = (20,20),
                            section_shift_fill_value = (255,255,255)
                            )
        return sectionshift(self.image), self.sectionshift_.__func__.__name__
        
    def sectionshift1_(self):
        sectionshift1 = SectionShift()
        return sectionshift1(self.image), self.sectionshift1_.__func__.__name__
        
        
        
    def squish_(self):
        squish = Squish(squish_direction = 1,
                squish_location = "random",
                squish_number_range = (5,10),
                squish_distance_range = (5,7),
                squish_line = "random",
                squish_line_thickness_range = (1,1)
                )
        return squish(self.image), self.squish_.__func__.__name__
        
    def squish1_(self):
        squish1 = Squish()
        return squish1(self.image), self.squish1_.__func__.__name__
        
        
        
        

    

