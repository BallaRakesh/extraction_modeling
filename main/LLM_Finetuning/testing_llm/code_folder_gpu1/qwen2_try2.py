from transformers import AutoModelForCausalLM, AutoTokenizer
# coder model !!
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

model_name = "Qwen/Qwen2-7B-Instruct"
# model_name = "Qwen/Qwen2-7B"
model_name = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ocr_data = """
        !\nMEDITERRANEAN SHIPPING COMPANY S.A.\n3=well or large version of the reverse | Ver p\u00e1gina Web per terminos y condiciones | CHOTOMTO B\u04355-DNA CEN\u00cdKOMM\u0435\u043d\u043b\u044f\u0441 \u0443\u0441\u043b\u043e\u0438\u043c\u0435 | www.mscmedshipoo.com\n**Fon-to-Port or \"Combined\nTransport\" (see Clause 1)\nBILL OF LADING No. MSCUN7511131\nWebsite: www.scmedshipco.com\nSCAC Code: MSCU\n2/3\nNO. OF RIDER PAGES\n\u0e04\nSHIPPER:\nMILLENNIUM METAL TRADING LLC\nP.O. BOX 64271 SHARJAH U.A.E\nCONSIGNEE: This B.L is not negotiablo unless marked \"To Order\" or \"To Order of...\u201d here,\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T. ROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\n\"NOTIFY PARTIES (No responsibility shall attach to the Carrier or to his Agent for failure to notify -\n[sec Clause 2D)\nMET TRADE INDIA LIMITED\nVILLAGE BHEEL AKBARPUR,\nG.T.\nROAD DADRI U.P. INDIA\nI.E. CODE: 0596067721\nORIGINAL\nNO. & SEQUENCE OF ORIGINAL B/L's\nCARRIER'S AGENTS ENDORSEMENTS: (Include Agent(s) at POD)\nFCL/FCL\nLloyds IMO Number =\nPORT OF DISCHARGE AGENT\nMSC GANDHIDHAM\n9108374\nMSC AGENCY (INDIA) PVT. LTD Siddhi Vinayak\nComplex, Plot 1,201-208,2 Fl. Junct.Tagore &\nAerodrome Rd, (W Side), In DC7, Ward 6\nTel:+91 2836 619129, Fax: +91 2836 619200\nEMAIL: gandhidham@mscindia.com\nVESSEL & VOYAGE NO. (see Clauses 8 & 9)\nKING JUSTUS V. WE301R\nBOOKING REF\n(or)\n775MXA1481..\nPORT OF LOADING\nNOUAKCHOTT\nSHIPPER'S REF. PORT OF DISCHARGE\nXXXXX MUNDRA\nPLACE OF RECEIPT: (Combined Transport ONLY -see Clauses 1 & 5.2)\nXXXXXXXX\nPLACE OF DELIVERY: (Combined Transport ONLY - see Clauses 1 & 5.2)\nDADRI\nPARTICULARS\nContainer Numbers, Sea!\nNumbers and Marke\n(Continued on attached Bill of Lading Rider page(s), if applicable)\n3X20' CNTR(S) S.T.C\nFURNISHED\nBY THE SHIPPER NOT CHECKED BY CARRIER CARRIER NOT RESPONSIBLE (see Clause 14)\nDescription of Packages and Goods\nGroes Cargo\nWeight\nKGS\nMeasurement\nSHIPPER'S LOAD STOW COUNT\nSCRAP CONFIRMING TO ISRI RAINS\n10 DAYS FREE TIME AT DESTINATION\n-SHIPPER'S LOAD STOW AND COUNT FOR\nINLAND HAULAGE CHARGES & DESTINATION HANDLING\nCHARGES ON CONSINEE'S ACCOUNT\nGLDU3430330/20DV\n1 NE\n24510.000\nCARRIER SEAL/67052\nMSCU3750155/20DV\n1 NE\n24480.000\nCARRIER SEAL/67066\nMSCU1447535/20DV\n1 NE\n24400.000\nCARRIER SEAL/57069\nTotal No. of Items 3 Total Gross wgt. 73390.000 KGS\nFREIGHT & CHARGES\nCargo shall not be delivered unless Freight & Charges are paid (see Clause 16).\nCertified True Copy\nSIGHT PREPAID\nDECLARED VALUE (only applicable If Ad Valorem\nCharges Doid - see Clause 7.3)\nXXXXX\nPLACE AND DATE OF ISSUE\nDUBAI, UAE 02-JAN-2013\n515\nStandard Edition - 06/2009\nFor Met Trade (India)\nRKg\nAuthorise\nRECEIVED by the Carrier In apparent good order and condition (unless otherwise\nstated herein) the total number or quantity of Containers or other packages or unit\nIndicated in the box entled Camer's Receipt for carriage subject to all the terms\nDischarge or Place of Delivery, whichever is applicable. IN ACCEPTING THIS BILL.\nOF LADING THE MERCHANT EXPRESSLY ACCEPTS AND AGREES TO ALL\nTHE TERMS AND CONDITIONS, WHETHER PRINTED, STAMPED OR\nOTHERWISE INCORPORATED ON THIS SIDE AND ON THE REVERSE SIDE OF\nBILL OF LADING AND THE TERMS AND CONDITIONS OF THE\nMERCHANT.\nAuthorised SignatER'S APPLICABLE TARIFF AS IF THEY WERE ALL SIGNED BY THE\nCARRIER'S RECEIPT (No. of Cntra or Pkgs rcvd by\nCarrier-see Clause 14.1\n3 CNTRS\nSHIPPED ON BOARD DATE\n02-JAN-2013\nIf this is a nagotable (To Order/of) Bill of Lading, one original Bill of Lading, duly\nEndorsed must be surrendered by the Merchant to the Carrier (together with\noutstanding Freight and charges) in exchange for the Goods or a Delivery Order. It\nthis is a non-negotiable (straight) Bill of Lading, the Carrier shall deliver the Goods\nor issue Delivery Order (after payment of outstanding Freight and charger)\nagainst the surrender of one original Bill of Lading or in accordance with the\nnational law at the Port of Discharge or Place of Delivery whichever is applicable.\nIN WITNESS WHEREOF the Carrier or their Agent has signed the number of Bilk\nof Lading stated at the top, all of this tanor and data, and wherover one original BIN\nof Lading has been surrendered all other Bills of Lading shall be void.\nS16N50.00 behalf of the Carrier MSC Mediterranean Shipping Company S.A\nFOMEDITERRANEAN SHIPPING COMPANY (U.A.E. (LL.C)\nAS AGENTS ON BEHALF OF THE CARRIER,\nTERMS CONTINUED ON REVERSE JMEDITERRANEAN SHIPPING SA GENEVA
        """
ocr_data = """
            Reliance Industries Limited EXTRA COPY NOT FOR CENVAT REMOVAL OF EXCISABLE GOODS FROM A FACTORY ( UNDER RULES 8,11,18,19 & 20 OF C.EX.RULES 2002 ) CTH No .: 3817 00 11 Exemption Notification No. & Date : Tariff Rate Consignee : A.R SULPHONATES PVT.LTD N - 41 ADDITIONAL AMBERNATH MIDC AREA ANANDNAGAR AMBERNATH 421506 Reliance Industries Limited B - 4 MIDC Area , P.O. Rasayani , Dist - Raigad , Patalganga - 410207 CEx . Regn . No .: AAACR5055KXM002 VAT TIN : 27110386481V W.E.F.01.04.2006 CST TIN : 27110386481C W.E.F.01.04.2006 ( 0030075855 ) TEL : 02512620191 INVOICE cum CHALLAN LST No .: / TIN No : 27360379214V WEF 01/04/2006 CST No .: 27360379214C WEF 01/04/2006 SNO Item Description Seal No 116157 / 8-8 Gross Wt Tare Wt Net Wt Name of Excisable Commodity : LINEAR ALKYL BENZENE 01 LINEAR ALKYL BENZENE FPG72 24.610 8.660 15.950 Name : Total Amount of Excise Duty : Rs . NIL Total Value of Goods Rs . ( in Figures ) : 2125593 / Hundred Ninety Three only CN No. PG00059604 Date : 18.09.2013 : MV No : GJ6W9998 Date of Issue of Invoice : 18.09.2013 Date and Time of Removal : 18.09.2013 15:35 Hrs . DCPI No .: 101054061 Buyer : N - 41 ADDITIONAL AMBERNATH MIDC AREA A.R SULPHONATES PVT.LTD MODE OF TRANSPORT Road despatch Truck Operator : EX - SERVICE MEN TRANSPORT CO Agent : Goods TransportAgency : FINE TECH CORPORATION PVT LTD Receiver's Signature : ANANDNAGAR AMBERNATH 421506 Batch No. & Descr . Qty ( MT ) Rate ( Rs./MT ) Amount ( Rs . ) No. of Pkgs . LST No. : / TIN No : 27360379214V WEF 01/04/2006 CST No. : 27360379214C WEF 01/04/2006 OrderNo : 1621753584 : 03 / 91 / 040 / 00396 / AM13 Dated 24/12/2012 SAMPLE BOTTLE GIVEN ( OTY . INCLUDED IN GROSS WEIGHT ) ADV INT . FILE NO & DTJt.DGFT File No CUST LICENCE NO W.DATECT3 NO . - ARSPL / 13-14 / CT - 3 / 019 Dated 12 th Sept 2013 CONTRACT REF.NO WITH DATELAB / 2012-2013 / ARSULPH / 001 / H Dated 6 th September Annex - B No. : 652 / 13 - 14 AGAINST CT3 NO . ASPL / CT - 3 / 019 Bulk Total Assessable Value CENVAT Notification No.22 / 2003 CE Dtd.31.03.2003 Duties have been rounded off in terms of section 37D of Central Excise Act , 1944 . Education Cess Sec.Hi. Edu . Cess LST / VAT Freight @ Rs.580.00 / MT 15.950 ** TAX INVOICE ** No. 126340.00 We hereby certify that our registration certificate under the Maharashtra Value Added Tax Act , 2002 is in force on the date on which the sale of goods specified in this tax invoiceis made by us and that the transaction of sale covered by this tax invoice has been effected by us and it shall be accounted for in the turnover of sales while filing of return andthe due tax , if any , payable on the sale has been paid or shall be paid \" . 12.00 % 2 % 1 % 5.00 % Authorised 400021 ( in words ) : Twenty One Lakh Twenty Five Thousand Five 2015123 2015123 LTU Membership No. LTU / MUM / 1112 RANGE The Superintendent , Central Excise & Service Tax , GLT - 3 , LTU Mumbai , 28th Floor , Centre - 1 , World Trade Centre , Cuffe Parade , Mumbai 400 005 . DIVISION : The Assistant Commissioner , Central Excise & Service Tax , LTU Mumbai , 28th Floor , Centre - 1 , World Trade Centre , Cuffe Parade , Mumbai - 400 005 . For Reliance Industries Limited A RELIANCE inator Checked By Regd.Office : 3rd Floor , Maker Chambers IV 222 , Nariman Point Mumbai The normal terms governing the above sale are printed overleaf . MUME & OE Declaration : Certified that all the particulars given above are true and correct . The Qunte indicated represents the price actually charged and there is no flow of additional consideration from the Buyer . C 101219 9251 NAVI 0 0 0 indirectly
        """
        
ocr_data = """
        Exporter M / S SMS EXPORTS G - 13 SKYLARK BUILDING , 60 NEHRU PLACE NEW DELHI - 110 019. ( INDIA ) AEPC / REG / MFG / 100652 TIN NO .: 07540195339 Consignee . M / S VILA A / S STORSKOVVEJ 20 ORMSLEV DK - 8260 VIBY J DENMARK Pre - carriage by Vessel / Flight No. Port of Discharge Marks & Nos / Container No. 55 PKGS . NOS . 01 TO 55 BY AIR No. & Kind of Pkgs Place of Receipt by Pre - carrier Port of Loading Dalkarlanabiksesoriasiaink IN - DEL - 04 Final Destination DENMARK INVOICE Description of Goods LADIES DRESS POLYESTER POWERLOOM READYMADE GARMENTS . Invoice No. & Date . S AIRWAY BILL NO .: 724-1081-5943 DT : 11.08.2012 Buyer's Order No. & Date SMSAVN936841 # 14012837 DRESEN DRESS ( # 1 - J. P. RETAIL & # 2 , # 3 - J. P. WHOLESALE ) Other Reference ( S ) IEC CODE NO . 0590020919 RBI CODE NO . DS - 007899 Buyer ( if other than consignee ) Country of Origin of Goods INDIA LESS DISCOUNT @ 2.5 % ON FOB Quantity Declaration We Declare that this Invoice shows the actual price of the goods described and that all particulars are true and correct . Export under chapter 3 of FTP . PACKING ID 160279501 PCS SMS / 529 / 2012-13 DT : 21.07.2012 1320 Amount Chargeable ( In words ) Total FOB Value In USDollar Fourteen Thousand Twenty Eight and Thirty Cents Only . QTY .: ONE THOUSAND THREE HUNDRED TWENTY PCS . ONLY . Country of Final Destination DENMARK Signature & Date Rate US $ 10.90 Amount FOB US $ 14,388.00 359.70 14,028.30 For S.M. S. EXPORTS Authorised Signatory
        """
        
prompt1 = f"""
    You are an expert at Extraction of data form Trade Finance Domain
    Here are the documet name:
    "Commercial Invoice"
    here is the ocr_data: {ocr_data}
    extract the [awb date,awb number, port of discharge]  from the ocr data
    Response:
    """

prompt4 = f"""
        You are an expert at Extraction of data form Trade Finance Domain
        Here are the documet name:
        Bill of Lading
        
        here is the ocr_data: {ocr_data}
        Extract the values for the list of keys from the ocr data
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
        remember the responce 
        """

messages = [
    {"role": "system", "content": "You are an expert at Extraction of data form Trade Finance Document"},
    {"role": "user", "content": prompt1}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2000
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(response)