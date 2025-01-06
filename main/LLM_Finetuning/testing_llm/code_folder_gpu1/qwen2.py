import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
model_path = 'openlm-research/open_llama_7b'

model_path = 'lmsys/vicuna-13b-v1.3'
# model_path = 'openlm-research/open_llama_13b'
model_path = "Qwen/Qwen2-7B"
model_path = '/home/gpu1admin/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e'

model_path = 'Qwen/Qwen2-1.5B'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.float16, device_map='auto',
# )
model = LlamaForCausalLM.from_pretrained(
    model_path, device_map='auto',
)
device = model.device  # This gets the device where the model is loaded (e.g., 'cuda:0')

ocr_data = """
        !\nMEDITERRANEAN SHIPPING COMPANY S.A.\n3=well or large version of the reverse | Ver p\u00e1gina Web per terminos y condiciones | CHOTOMTO B\u04355-DNA CEN\u00cdKOMM\u0435\u043d\u043b\u044f\u0441 \u0443\u0441\u043b\u043e\u0438\u043c\u0435 | www.mscmedshipoo.com\n**Fon-to-Port or \"Combined\nTransport\" (see Clause 1)\nBILL OF LADING No. MSCUN7511131\nWebsite: www.scmedshipco.com\nSCAC Code: MSCU\n2/3\nNO. OF RIDER PAGES\n\u0e04\nSHIPPER:\nMILLENNIUM METAL TRADING LLC\nP.O. BOX 64271 SHARJAH U.A.E\nCONSIGNEE: This B.L is not negotiablo unless marked \"To Order\" or \"To Order of...\u201d here,\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T. ROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\n\"NOTIFY PARTIES (No responsibility shall attach to the Carrier or to his Agent for failure to notify -\n[sec Clause 2D)\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T.\nROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\nORIGINAL\nNO. & SEQUENCE OF ORIGINAL B/L's\nCARRIER'S AGENTS ENDORSEMENTS: (Include Agent(s) at POD)\nFCL/FCL\nLloyds IMO Number =\nPORT OF DISCHARGE AGENT\nMSC GANDHIDHAM\n9108374\nMSC AGENCY (INDIA) PVT. LTD Siddhi Vinayak\nComplex, Plot 1,201-208,2 Fl. Junct.Tagore &\nAerodrome Rd, (W Side), In DC7, Ward 6\nTel:+91 2836 619129, Fax: +91 2836 619200\nEMAIL: gandhidham@mscindia.com\nVESSEL & VOYAGE NO. (see Clauses 8 & 9)\nKING JUSTUS V. WE301R\nBOOKING REF\n(or)\n775MXA1481..\nPORT OF LOADING\nNOUAKCHOTT\nSHIPPER'S REF. PORT OF DISCHARGE\nXXXXX MUNDRA\nPLACE OF RECEIPT: (Combined Transport ONLY -see Clauses 1 & 5.2)\nXXXXXXXX\nPLACE OF DELIVERY: (Combined Transport ONLY - see Clauses 1 & 5.2)\nDADRI\nPARTICULARS\nContainer Numbers, Sea!\nNumbers and Marke\n(Continued on attached Bill of Lading Rider page(s), if applicable)\n3X20' CNTR(S) S.T.C\nFURNISHED\nBY THE SHIPPER NOT CHECKED BY CARRIER CARRIER NOT RESPONSIBLE (see Clause 14)\nDescription of Packages and Goods\nGroes Cargo\nWeight\nKGS\nMeasurement\nSHIPPER'S LOAD STOW COUNT\nSCRAP CONFIRMING TO ISRI RAINS\n10 DAYS FREE TIME AT DESTINATION\n-SHIPPER'S LOAD STOW AND COUNT FOR\nINLAND HAULAGE CHARGES & DESTINATION HANDLING\nCHARGES ON CONSINEE'S ACCOUNT\nGLDU3430330/20DV\n1 NE\n24510.000\nCARRIER SEAL/67052\nMSCU3750155/20DV\n1 NE\n24480.000\nCARRIER SEAL/67066\nMSCU1447535/20DV\n1 NE\n24400.000\nCARRIER SEAL/57069\nTotal No. of Items 3 Total Gross wgt. 73390.000 KGS\nFREIGHT & CHARGES\nCargo shall not be delivered unless Freight & Charges are paid (see Clause 16).\nCertified True Copy\nSIGHT PREPAID\nDECLARED VALUE (only applicable If Ad Valorem\nCharges Doid - see Clause 7.3)\nXXXXX\nPLACE AND DATE OF ISSUE\nDUBAI, UAE 02-JAN-2013\n515\nStandard Edition - 06/2009\nFor Met Trade (India)\nRKg\nAuthorise\nRECEIVED by the Carrier In apparent good order and condition (unless otherwise\nstated herein) the total number or quantity of Containers or other packages or unit\nIndicated in the box entled Camer's Receipt for carriage subject to all the terms\nDischarge or Place of Delivery, whichever is applicable. IN ACCEPTING THIS BILL.\nOF LADING THE MERCHANT EXPRESSLY ACCEPTS AND AGREES TO ALL\nTHE TERMS AND CONDITIONS, WHETHER PRINTED, STAMPED OR\nOTHERWISE INCORPORATED ON THIS SIDE AND ON THE REVERSE SIDE OF\nBILL OF LADING AND THE TERMS AND CONDITIONS OF THE\nMERCHANT.\nAuthorised SignatER'S APPLICABLE TARIFF AS IF THEY WERE ALL SIGNED BY THE\nCARRIER'S RECEIPT (No. of Cntra or Pkgs rcvd by\nCarrier-see Clause 14.1\n3 CNTRS\nSHIPPED ON BOARD DATE\n02-JAN-2013\nIf this is a nagotable (To Order/of) Bill of Lading, one original Bill of Lading, duly\nEndorsed must be surrendered by the Merchant to the Carrier (together with\noutstanding Freight and charges) in exchange for the Goods or a Delivery Order. It\nthis is a non-negotiable (straight) Bill of Lading, the Carrier shall deliver the Goods\nor issue Delivery Order (after payment of outstanding Freight and charger)\nagainst the surrender of one original Bill of Lading or in accordance with the\nnational law at the Port of Discharge or Place of Delivery whichever is applicable.\nIN WITNESS WHEREOF the Carrier or their Agent has signed the number of Bilk\nof Lading stated at the top, all of this tanor and data, and wherover one original BIN\nof Lading has been surrendered all other Bills of Lading shall be void.\nS16N50.00 behalf of the Carrier MSC Mediterranean Shipping Company S.A\nFOMEDITERRANEAN SHIPPING COMPANY (U.A.E. (LL.C)\nAS AGENTS ON BEHALF OF THE CARRIER,\nTERMS CONTINUED ON REVERSE JMEDITERRANEAN SHIPPING SA GENEVA
        """

# ocr_data = """
# 67066\nMSCU1447535/20DV\n1 NE\n24400.000\nCARRIER SEAL/57069\nTotal No. of Items 3 Total Gross wgt. 73390.000 KGS\nFREIGHT & CHARGES\nCargo shall not be delivered unless Freight & Charges are paid (see Clause 16).\nCertified True Copy\nSIGHT PREPAID\nDECLARED VALUE (only applicable If Ad Valorem\nCharges
# """
prompt = f"""
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
    3) The genearated address should be in different patters
    Response:
    """

prompt1 = f"""
    You are an expert at Extraction of data form Trade Finance Domain
    Here are the documet name:
    "Bill of Lading"
    here is the ocr_data: {ocr_data}
    extract the gross weight from the ocr data
    Response:
    """

def generate_responce(prompt_):
    # prompt_ = 'Hi How are you'
    inputs = tokenizer(prompt_, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    inputs = inputs.to(device)
    output = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1, temperature=0.0)
                # do_sample=True,top_k=50,top_p=0.95, max_new_tokens=2000, max_length=2000
                
    generated_texts = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    
    return generated_texts

res = generate_responce(prompt1)
print(res)