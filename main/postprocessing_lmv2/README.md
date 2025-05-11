# Report_Generation_for_lmv2
## NOTE:
## CHANGES NEED TO MAKE IN CONFIG FILE
### 1. /post_processing/config/config.ini
       mention the document names for generating the reports
        eg : document_code = [ic]
### 2. /post_processing/config/prod.ini
       change the paths here according to the mentioned Document Name 
        eg:
        [TransportDocument]
        BillOfLadding = /home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/BOL

### 3. post_processing/post_processing/config.ini
       i-> provide the document wise post processing json files
        eg:
        [TransportDocument]
        bol = /home/ntlpt19/Downloads/Evaluation_Data/updated_code/post_processing/PP_files/   BOL_post_process.json
      ii -> provide the rules_required_keys, most_frequently_occurring_keys json files 
        eg: 
        [PATH]
        rules_required_keys = /home/ntlpt19/Downloads/Evaluation_Data/rule_documents_keys.json
      iii -> mention the category wise fuzzy ratios
        eg:
        [FuzzyRatio]
        numeric_fields = 80

      iV -> mention the lookup paths 
        eg:
        [LookUp]
        incoterms = /home/ntlpt19/Downloads/Evaluation_Data/updated_code/post_processing post_processing/incoterm_list.txt
    
### 4. check premapping here /post_processing/pre_mapping.py
       i-> PredictionKeyMapping
       ii-> ParentKeyMapping
       iii-> OverLappingKeys
       iv -> top_value
       v -> bottom_value
       vi -> list of category wise fields
       
 
    

    
