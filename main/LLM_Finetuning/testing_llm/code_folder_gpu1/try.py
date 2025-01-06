from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
# from Batch_data_prep import generate_instruction, generate_prompt, generate_prompt_inference
import os
import pandas as pd
import time
import re
import json
import torch
from transformers import BitsAndBytesConfig

import torch
from transformers import BitsAndBytesConfig


low_performance_keys = ["drawee address",
                        "drawee name",
                        "drawer address",
                        "invoice amount",
                        "invoice currency:",
                        "invoice date",
                        "invoice number",
                        "issuing bank",
                        "issuing bank address",
                        "letter of credit reference number",
                        "original or copy",
                        "tenor type"
]

ocr_data = """
        !\nMEDITERRANEAN SHIPPING COMPANY S.A.\n3=well or large version of the reverse | Ver p\u00e1gina Web per terminos y condiciones | CHOTOMTO B\u04355-DNA CEN\u00cdKOMM\u0435\u043d\u043b\u044f\u0441 \u0443\u0441\u043b\u043e\u0438\u043c\u0435 | www.mscmedshipoo.com\n**Fon-to-Port or \"Combined\nTransport\" (see Clause 1)\nBILL OF LADING No. MSCUN7511131\nWebsite: www.scmedshipco.com\nSCAC Code: MSCU\n2/3\nNO. OF RIDER PAGES\n\u0e04\nSHIPPER:\nMILLENNIUM METAL TRADING LLC\nP.O. BOX 64271 SHARJAH U.A.E\nCONSIGNEE: This B.L is not negotiablo unless marked \"To Order\" or \"To Order of...\u201d here,\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T. ROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\n\"NOTIFY PARTIES (No responsibility shall attach to the Carrier or to his Agent for failure to notify -\n[sec Clause 2D)\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T.\nROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\nORIGINAL\nNO. & SEQUENCE OF ORIGINAL B/L's\nCARRIER'S AGENTS ENDORSEMENTS: (Include Agent(s) at POD)\nFCL/FCL\nLloyds IMO Number =\nPORT OF DISCHARGE AGENT\nMSC GANDHIDHAM\n9108374\nMSC AGENCY (INDIA) PVT. LTD Siddhi Vinayak\nComplex, Plot 1,201-208,2 Fl. Junct.Tagore &\nAerodrome Rd, (W Side), In DC7, Ward 6\nTel:+91 2836 619129, Fax: +91 2836 619200\nEMAIL: gandhidham@mscindia.com\nVESSEL & VOYAGE NO. (see Clauses 8 & 9)\nKING JUSTUS V. WE301R\nBOOKING REF\n(or)\n775MXA1481..\nPORT OF LOADING\nNOUAKCHOTT\nSHIPPER'S REF. PORT OF DISCHARGE\nXXXXX MUNDRA\nPLACE OF RECEIPT: (Combined Transport ONLY -see Clauses 1 & 5.2)\nXXXXXXXX\nPLACE OF DELIVERY: (Combined Transport ONLY - see Clauses 1 & 5.2)\nDADRI\nPARTICULARS\nContainer Numbers, Sea!\nNumbers and Marke\n(Continued on attached Bill of Lading Rider page(s), if applicable)\n3X20' CNTR(S) S.T.C\nFURNISHED\nBY THE SHIPPER NOT CHECKED BY CARRIER CARRIER NOT RESPONSIBLE (see Clause 14)\nDescription of Packages and Goods\nGroes Cargo\nWeight\nKGS\nMeasurement\nSHIPPER'S LOAD STOW COUNT\nSCRAP CONFIRMING TO ISRI RAINS\n10 DAYS FREE TIME AT DESTINATION\n-SHIPPER'S LOAD STOW AND COUNT FOR\nINLAND HAULAGE CHARGES & DESTINATION HANDLING\nCHARGES ON CONSINEE'S ACCOUNT\nGLDU3430330/20DV\n1 NE\n24510.000\nCARRIER SEAL/67052\nMSCU3750155/20DV\n1 NE\n24480.000\nCARRIER SEAL/67066\nMSCU1447535/20DV\n1 NE\n24400.000\nCARRIER SEAL/57069\nTotal No. of Items 3 Total Gross wgt. 73390.000 KGS\nFREIGHT & CHARGES\nCargo shall not be delivered unless Freight & Charges are paid (see Clause 16).\nCertified True Copy\nSIGHT PREPAID\nDECLARED VALUE (only applicable If Ad Valorem\nCharges Doid - see Clause 7.3)\nXXXXX\nPLACE AND DATE OF ISSUE\nDUBAI, UAE 02-JAN-2013\n515\nStandard Edition - 06/2009\nFor Met Trade (India)\nRKg\nAuthorise\nRECEIVED by the Carrier In apparent good order and condition (unless otherwise\nstated herein) the total number or quantity of Containers or other packages or unit\nIndicated in the box entled Camer's Receipt for carriage subject to all the terms\nDischarge or Place of Delivery, whichever is applicable. IN ACCEPTING THIS BILL.\nOF LADING THE MERCHANT EXPRESSLY ACCEPTS AND AGREES TO ALL\nTHE TERMS AND CONDITIONS, WHETHER PRINTED, STAMPED OR\nOTHERWISE INCORPORATED ON THIS SIDE AND ON THE REVERSE SIDE OF\nBILL OF LADING AND THE TERMS AND CONDITIONS OF THE\nMERCHANT.\nAuthorised SignatER'S APPLICABLE TARIFF AS IF THEY WERE ALL SIGNED BY THE\nCARRIER'S RECEIPT (No. of Cntra or Pkgs rcvd by\nCarrier-see Clause 14.1\n3 CNTRS\nSHIPPED ON BOARD DATE\n02-JAN-2013\nIf this is a nagotable (To Order/of) Bill of Lading, one original Bill of Lading, duly\nEndorsed must be surrendered by the Merchant to the Carrier (together with\noutstanding Freight and charges) in exchange for the Goods or a Delivery Order. It\nthis is a non-negotiable (straight) Bill of Lading, the Carrier shall deliver the Goods\nor issue Delivery Order (after payment of outstanding Freight and charger)\nagainst the surrender of one original Bill of Lading or in accordance with the\nnational law at the Port of Discharge or Place of Delivery whichever is applicable.\nIN WITNESS WHEREOF the Carrier or their Agent has signed the number of Bilk\nof Lading stated at the top, all of this tanor and data, and wherover one original BIN\nof Lading has been surrendered all other Bills of Lading shall be void.\nS16N50.00 behalf of the Carrier MSC Mediterranean Shipping Company S.A\nFOMEDITERRANEAN SHIPPING COMPANY (U.A.E. (LL.C)\nAS AGENTS ON BEHALF OF THE CARRIER,\nTERMS CONTINUED ON REVERSE JMEDITERRANEAN SHIPPING SA GENEVA
        """


# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_path = '/home/gpu1admin/rakesh/code/model'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)#, quantization_config=model_quantization())
# # save_model(model_path, model, tokenizer)

model_path = '/datadrive/MistralModels/Base_Model/model_base'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_responce(prompt_):
    
    inputs = tokenizer(prompt_, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    output = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1, temperature=0.0)
                # do_sample=True,top_k=50,top_p=0.95, max_new_tokens=2000, max_length=2000
                
    generated_texts = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    
    return generated_texts


prompt1 = f"""
    You are an expert at generating synthetic data for Trade Finance Domain
    Here are the documet name:
    "Bill of Exhange"

    Here's the sample Example for the 'drawer address':
    'SCO.9-11 SECTOR - 9D MADHYA MARG CIBD CHANDIGARH 160017 INDIA'
    'EMPIRE HOUSE FLOOR 1'
    '29 , balanjaneya temple street , opp . m.s.ramaiah hospital , kge layout , r.m.v. ii stage , bangalore - 560 094. india'
    '34a , metcalfe street , 3rd floor , kolkata - 700 013 , india ,'
    \n
    By looking at the above Example please generate the atleast five different drawer addresses 
    Instructions need to follow:
    1) the addresses should be completely different from each other 
    2) the addresses should be complete address, which should me max length
    3) By looking at the above Example please generate the new address
    4) The genearated address should be in different patters
    Response:
    """


prompt2 = f"""
    You are an expert at generating synthetic data for Trade Finance Domain
    Here are the documet name:
    "Bill of Exhange"

    Here's the sample Example:
    {ocr_data}
    By looking at the above Example please generate the similar kind of data
    Response:
    """

prompt3 = f"""
    You are an expert at generating synthetic data for Trade Finance Domain
    Here are the documet name:
    "Bill of Lading"
    You Need to Generate the synthetic OCR data using these list of keys:
    [
        "shipper_name": "PANTHEON FZE",
        "shipper_address": "P.O. BOX 17899 , JEBEL ALI , DUBAI , UNITED ARAB EMIRATES",
        "consignee_name": "SAMRAT PHARMACHEM LIMITED",
        "consignee_address": "PLOT NO . A2 / 3445 , GIDC , PHASE 4 ANKLESHWAR - 393 002 , GUJARAT , INDIA .",
        "notify_party_name": "SAMRAT PHARMACHEM LIMITED",
        "notify_party_address": "PLOT NO . A2 / 3445 , GIDC , PHASE 4 ANKLESHWAR - 393 002 , GUJARAT , INDIA .",
        "place_of_receipt": "IQUIQUE , CHILE",
        "vessel_name": "NYK TERRA",
        "voyage_number": "0308W",
        "port_of_loading": "IQUIQUE , CHILE",
        "port_of_discharge": "JHAVA SHEVA , INDIA",
        "bill_of_lading_number": "MOLU27801430859",
        "container_number": "CXDU1638114 / S2",
        "goods_description": "CRUDE IODINE 99.5 % CHILE ORIGIN",
        "gross_weight": "10,280.000",
        "net_weight": "10,000.000",
        "bol_original_number": "THREE",
        "place_of_issue": "DUBAI",
        "bill_of_lading_issue_date": "24-12-2012",
        "freight_collect_or_prepaid": "FREIGHT PREPAID",
        "final_destination": "NHAVA SHEVA . INDIA",
        "carrier_name": "Mitsui O.S.K. Lines , Ltd.",
        "bol_original_or_copy": "COPY",
        "dimension": "200 X 50",
        "goods_marks_and_nos": "200 DRUMS",
        "measurement": "30.000",
        "agent_name": "MITSUI OSK LINES INDIA PVT . LTD .",
        "agent_country": "GR . FLOOR , PORT USER'S BLDG . , J.N.PORT NHAVASHEVA , NAVI MUMBAI- 400070 , MAHARASHTRA"
        ]

    
    Here's the sample Example:
    {ocr_data}
    this OCR data is generated from these list of keys 
    [
    "shipper_name": "MILLENNIUM METAL TRADING LLC",
    "shipper_address": "P.O. BOX 64271 SHARJAH U.A.E",
    "consignee_name": "MET TRADE INDIA LIMITED",
    "consignee_address": "VILLAGE BHEEL AKBARPUR , G.T. ROAD DADRI U.P. INDIA",
    "notify_party_name": "MET TRADE INDIA LIMITED",
    "notify_party_address": "VILLAGE BHEEL AKBARPUR , G.T. ROAD DADRI U.P. INDIA",
    "vessel_name": "KING JUSTUS",
    "voyage_number": "V. WE301R",
    "port_of_loading": "NOUAKCHOTT",
    "port_of_discharge": "MUNDRA",
    "container_number": [
        "MSCU3750155 / 20DV",
        "GLDU3430330 / 20DV",
        "MSCU1447535 / 20DV"
    ],
    "gross_weight": "73390.000 KGS",
    "place_of_issue": "DUBAI , UAE",
    "shipped_onboard_date": "02 - JAN - 2013",
    "signed_By_agent": "has been surrendered all S16N50.00 behalf of the Carrier MSC Mediterranean FOMEDITERRANEAN SHIPPING COMPANY ( U.A.E. AGENTS ON BEHALF OF THE CARRIER JMEDITERRANEAN SHIPPING SA",
    "freight_collect_or_prepaid": "SIGHT PREPAID",
    "bill_of_lading_number": "** Fon - to - Port MSCUN7511131",
    "agent_name": "MSC AGENCY ( INDIA ) PVT . LTD",
    "agent_country": [
        "Siddhi Vinayak",
        "Complex , Plot 1,201-208,2 Fl . Junct.Tagore Aerodrome Rd , ( W Side ) , In DC7 , Ward 6"
    ],
    "final_destination": "DADRI",
    "goods_description": "SCRAP CONFIRMING TO ISRI RAINS",
    "bol_original_or_copy": "ORIGINAL",
    "bill_of_lading_issue_date": "02 - JAN - 2013",
    "dimension": "3X20",
    "bol_original_number": "2/3 CARRIER'S AGENTS",
    "signature": "Trade ( ) RKg Authorise Discharge OF THE Authorised SignatER'S",
    "carrier_name": "MEDITERRANEAN SHIPPING COMPANY S.A.",
    "goods_marks_and_nos": "3 CNTRS"
    ]
    Response:
    """


prompt4 = f"""
        You are an expert at Extraction of data form Trade Finance Domain
        Here are the documet name:
        Bill of Lading
        
        here is the ocr_data: {ocr_data}
        Extract the list of keys from the ocr data
        Here is the list of Keys
        keys = [
            "shipper_name",
            "shipper_address",
            "consignee_name",
            "consignee_address",
            "notify_party_name",
            "notify_party_address",
            "vessel_name",
            "voyage_number",
            "port_of_loading",
            "port_of_discharge",
            "container_number",
            "gross_weight",
            "place_of_issue",
            "shipped_onboard_date",
            "signed_By_agent",
            "freight_collect_or_prepaid",
            "bill_of_lading_number",
            "agent_name",
            "agent_country",
            "final_destination",
            "goods_description",
            "bol_original_or_copy",
            "bill_of_lading_issue_date",
            "dimension",
            "bol_original_number",
            "signature",
            "carrier_name",
            "goods_marks_and_nos"
        ]
        Response:
        """
    
    
prompt5 = f"""
    You are an expert at Extraction of data form Trade Finance Domain
    Here are the documet name:
    "Bill of Lading"
    here is the ocr_data: {ocr_data}
    extract the gross weight from the ocr data
    Response:
    """

prediction = generate_responce(prompt5)
print('>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>')
print(prediction)