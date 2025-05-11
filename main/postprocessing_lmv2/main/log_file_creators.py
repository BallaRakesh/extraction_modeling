from src.main.extraction.utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
import datetime

set_basic_config_for_logging(folder_path = "src/main", 
                                 filename = f'''data_preparation_{"_".join(str(datetime.now()).split(" "))}''')
data_prep_logger = get_logger_object_and_setting_the_loglevel()


set_basic_config_for_logging(folder_path="src/main",
                             filename="training_utility")
train_utility_logger = get_logger_object_and_setting_the_loglevel()
