import os
from pdf2image import convert_from_path

class PDFsplitter:   
    def __init__(self,pdf_path,image_output_folder):
        self.output_folder = image_output_folder
        self.input_folder = pdf_path
    
    def pdf_to_images(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # Convert PDF to images
        images = convert_from_path(self.input_folder)
        # Save each image
        for i, image in enumerate(images):
            image.save(os.path.join(self.input_folder, f'page_{i+1}.png'), 'PNG')

if __name__ == '__main__':
    pdf_storing_path = 'ENTER THE PDF INPUT PATH'
    result_image_storing_path = 'ENTER THE DESTINATION PATH'
    pdf_splitter = PDFsplitter(pdf_storing_path,result_image_storing_path)
    pdf_splitter.pdf_to_images()
    print("=================================================================")
    print("SUCCESSFULLY FINISHED CONVERTING ALL PDF'S INTO INDIVIDUAL IMAGES")
    print("=================================================================")