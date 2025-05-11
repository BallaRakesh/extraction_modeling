

address_fields : list = ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                            "drawee_address", "consignee_address", "consignor_address", \
                                "coo_issuer_address", 'nostro_bank_address', 'consignor_address', \
                                    'address_of_assured', 'drawee_address', 'insurance_issuer_address', \
                                        'remitter_address', 'beneficiary_address', 'coo_issuer_address', \
                                            'notify_party_address', 'drawer_bank_address', 'consignee_address', \
                                                'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', \
                                                    'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address',\
                                                        'drawer_exporter_seller_beneficiary_issuer_supplier_address'\
                                                            'remitter_drawee_buyer_importer_billto_shippedto_despacthto_address'\
                                                                'beneficiary_bank_address', 'shipper_address', 'consignor_address', 'notify_party_address'\
                                                                    'consignee_address', 'drawer_exporter_seller_beneficiary_issuer_supplier_address'\
                                                                        'remitter_drawee_buyer_importer_billto_shippedto_despacthto_address']


date_fields : list =["awb_date","bill_of_lading_date","bill_of_lading_issue_date",\
                "csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date",\
                    "invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date",\
                        "shipped_onboard_date","start_date","tenor_indicator_date","transaction_date", 'invoice_date', 'invoice_due_date',\
                            'order_date', 'shipment_date', 'issue_date']


fuzzy_match_keys : list = ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address","page_no","csh_drawn_under_rules", \
                                            "doc_charge_instructions",
                                        "doc_delivery_instruction","csh_bill_currency",\
                                            "csh_presentation_date","csh_due_date", "consignee_address", \
                                                "consignor_address", "coo_issuer_address", 'nostro_bank_address', \
                                                    'consignor_address', 'address_of_assured', 'drawee_address', \
                                                        'insurance_issuer_address', 'remitter_address', 'beneficiary_address', \
                                                            'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', \
                                                                'consignee_address', 'drawer_address', 'drawee_bank_address', \
                                                                    'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', \
                                                                        'claim_payable_by_address']

keys_restrict_from_remove_spl_char : list = ["awb_date","bill_of_lading_date","bill_of_lading_issue_date",\
                                            "csh_presentation_date","date_of_invoice","end_date","expiry_date",\
                                                "indicator_date","invoice_date","invoice_due_date",\
                                                "issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date",\
                                                    "start_date","tenor_indicator_date","transaction_date", \
                                                    "awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date",\
                                                        "date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date",\
                                                            "lc_date","sail_on_or_about_to_date","shipped_onboard_date",\
                                                            "start_date","tenor_indicator_date","transaction_date", 'performa_invoice_no']


multi_liner_keys_for_merging : list =  ["drawee_bank_address", "drawer_bank_address",\
                                "drawee_address","drawer_address", "document_enclosed", "consignee_address","consignor_address", \
                                    "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages", 'notify_party_name','dimension', \
                                        'consignee_addres', 'carrier_country', 'carrier_name', 'agent_country','agent_name','nostro_bank_address', \
                                            'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', \
                                                'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', \
                                                    'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', \
                                                        'claim_payable_by_address', 'goods_description', 'terms_of_delivery', 'packaging_description', 'diclaration_by', 'delivery_terms', \
                                                            'item_description', 'packaging_details', 'dimension']

keys_not_required : list =  ['currency_amount',"drawee_bank_country", "drawee_country", "drawer_country","drawer_bank_country", 'signature', 'signed_stamp']


weights_fields : list = ['net_weight', 'gross_weight','total_quantity_of_goods', 'tolerance_of_quantity', 'quantity', 'unit', 'rate', 'net_weight', 'unit_price']

amount_fields : list = ['total_amount_in_numeric', 'tax_amount', 'amount_due', 'advance_amount', 'total_amount_in_figure', 'tax_amount']

currency_fields : list = ['currency_in_numeric', 'currency']

transport_fields : list = ['pre_carriage_by', 'mode_of_transport', 'means_of_transport', 'pre_carriage_by', 'mode_of_dispatch']