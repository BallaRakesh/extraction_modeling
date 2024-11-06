import os 

TRAIN_FLAG = True 

class TextFileGeneration:
    
    def __init__(self):
        self.training_folder_path = 'TRAIN FOLDER PATH'
        self.test_folder_path = 'TEST FOLDER PATH'
        self.train_line_list = []
        self.test_line_list = []
        self.train_list_length = 0
        self.test_list_length = 0 
    
    def train_list_generation(self):
        zero_deg_image_path = os.path.join(self.training_folder_path,'img_0')
        ninety_deg_image_path = os.path.join(self.training_folder_path,'img_90')
        one_eighty_deg_image_path = os.path.join(self.training_folder_path,'img_180')
        two_seventy_deg_image_path = os.path.join(self.training_folder_path,'img_270')
        
        result_path = os.path.join(self.training_folder_path,'train_list.txt')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        line_list = []
        if os.path.exists(zero_deg_image_path):
            for file in os.listdir(zero_deg_image_path):
                line_name = 'img_0/' + file + " 0" 
                file_path = os.path.join(zero_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(ninety_deg_image_path):
            for file in os.listdir(ninety_deg_image_path):
                line_name = 'img_90/' + file + " 1" 
                file_path = os.path.join(ninety_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(one_eighty_deg_image_path):
            for file in os.listdir(one_eighty_deg_image_path):
                line_name = 'img_180/' + file + " 2" 
                file_path = os.path.join(one_eighty_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(two_seventy_deg_image_path):
            for file in os.listdir(two_seventy_deg_image_path):
                line_name = 'img_270/' + file + " 3" 
                file_path = os.path.join(two_seventy_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        with open(result_path, 'w') as file:
            for line in line_list:
                file.write(line + '\n')
        self.train_list_length = len(line_list)
        
        
    def train_list_debug_generation(self):
        zero_deg_image_path = os.path.join(self.training_folder_path,'img_0')
        ninety_deg_image_path = os.path.join(self.training_folder_path,'img_90')
        one_eighty_deg_image_path = os.path.join(self.training_folder_path,'img_180')
        two_seventy_deg_image_path = os.path.join(self.training_folder_path,'img_270')
        
        num_files = int(self.train_list_length * 0.025)
        num_file_per_category = num_files // 4
        result_path = os.path.join(self.training_folder_path,'train_list.txt.debug')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        line_list = []
        
        if os.path.exists(zero_deg_image_path):
            counter = 0
            for file in os.listdir(zero_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_0/' + file + " 0" 
                file_path = os.path.join(zero_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        if os.path.exists(ninety_deg_image_path):
            counter = 0
            for file in os.listdir(ninety_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_90/' + file + " 1" 
                file_path = os.path.join(ninety_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        if os.path.exists(one_eighty_deg_image_path):
            counter = 0
            for file in os.listdir(one_eighty_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_180/' + file + " 2" 
                file_path = os.path.join(one_eighty_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)

        if os.path.exists(two_seventy_deg_image_path):
            counter = 0
            for file in os.listdir(two_seventy_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_270/' + file + " 3" 
                file_path = os.path.join(two_seventy_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        with open(result_path, 'w') as file:
            for line in line_list:
                file.write(line + '\n')   

    def test_list_generation(self):
        zero_deg_image_path = os.path.join(self.test_folder_path,'img_0')
        ninety_deg_image_path = os.path.join(self.test_folder_path,'img_90')
        one_eighty_deg_image_path = os.path.join(self.test_folder_path,'img_180')
        two_seventy_deg_image_path = os.path.join(self.test_folder_path,'img_270')
        
        result_path = os.path.join(self.test_folder_path,'test_list.txt')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        line_list = []
        if os.path.exists(zero_deg_image_path):
            for file in os.listdir(zero_deg_image_path):
                line_name = 'img_0/' + file + " 0" 
                file_path = os.path.join(zero_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(ninety_deg_image_path):
            for file in os.listdir(ninety_deg_image_path):
                line_name = 'img_90/' + file + " 1" 
                file_path = os.path.join(ninety_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(one_eighty_deg_image_path):
            for file in os.listdir(one_eighty_deg_image_path):
                line_name = 'img_180/' + file + " 2" 
                file_path = os.path.join(one_eighty_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        if os.path.exists(two_seventy_deg_image_path):
            for file in os.listdir(two_seventy_deg_image_path):
                line_name = 'img_270/' + file + " 3" 
                file_path = os.path.join(two_seventy_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        with open(result_path, 'w') as file:
            for line in line_list:
                file.write(line + '\n')
        
        self.test_line_list = line_list
        self.test_list_length = len(line_list)
        
    def test_list_debug_generation(self):
        zero_deg_image_path = os.path.join(self.test_folder_path,'img_0')
        ninety_deg_image_path = os.path.join(self.test_folder_path,'img_90')
        one_eighty_deg_image_path = os.path.join(self.test_folder_path,'img_180')
        two_seventy_deg_image_path = os.path.join(self.test_folder_path,'img_270')
        
        num_files = int(self.train_list_length * 0.025)
        num_file_per_category = num_files // 4
        result_path = os.path.join(self.test_folder_path,'test_list.txt.debug')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        line_list = []
        
        if os.path.exists(zero_deg_image_path):
            counter = 0
            for file in os.listdir(zero_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_0/' + file + " 0" 
                file_path = os.path.join(zero_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        if os.path.exists(ninety_deg_image_path):
            counter = 0
            for file in os.listdir(ninety_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_90/' + file + " 1" 
                file_path = os.path.join(ninety_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)
        
        if os.path.exists(one_eighty_deg_image_path):
            counter = 0
            for file in os.listdir(one_eighty_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_180/' + file + " 2" 
                file_path = os.path.join(one_eighty_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)

        if os.path.exists(two_seventy_deg_image_path):
            counter = 0
            for file in os.listdir(two_seventy_deg_image_path):
                counter += 1 
                if counter == num_file_per_category:
                    break 
                line_name = 'img_270/' + file + " 3" 
                file_path = os.path.join(two_seventy_deg_image_path,file)
                if os.path.exists(file_path):
                    line_list.append(line_name)

        with open(result_path, 'w') as file:
            for line in line_list:
                file.write(line + '\n')

if __name__ == '__main__':
    
    text_file_generator = TextFileGeneration()
    text_file_generator.train_list_generation()
    text_file_generator.train_list_debug_generation()
    text_file_generator.test_list_generation()
    text_file_generator.test_list_debug_generation()
