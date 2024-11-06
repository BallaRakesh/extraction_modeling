

from final_accuracy_gen import extract_currency_and_amount

def currency_amount_segregation(pred_actual_list, data_json, child_key1, child_key2, parent_field = None, prediction_flag = None):
    '''
        pred_actual_list =  list of keys in prediction dictnary or actual dictnary
        parent_field = "currency_amount"
        child_key1 = "csh_bill_amount"
        child_key2 = "csh_bill_currency"
        ##### child_key1 or child_key2 may overlap need to check properly before assigning ######
        
    '''
    
    if parent_field:
        if parent_field in pred_actual_list:
            for i in range(len(data_json[parent_field])):
                act_currency, act_amount = extract_currency_and_amount(str(data_json[parent_field][i][0]))
                if len(act_amount)>0 and child_key1 in pred_actual_list:
                    if prediction_flag:
                        data_json[child_key1].append([act_amount, data_json[parent_field][i][1], data_json[parent_field][i][2]])

                    else:
                        data_json[child_key1].append([act_amount, data_json[parent_field][i][1]])
                    # data_json[child_key1].append(data_json[parent_field][i][1])
                else:
                    if len(act_amount)>0:
                        if prediction_flag:
                            data_json[child_key1]=[[act_amount, data_json[parent_field][i][1], data_json[parent_field][i][2]]]
                        else:
                            data_json[child_key1]=[[act_amount, data_json[parent_field][i][1]]]


                if len(act_currency)>0 and child_key2 in pred_actual_list:
                    if prediction_flag:
                        data_json[child_key2].append([act_currency, data_json[parent_field][i][1], data_json[parent_field][i][2]])

                    else:
                        data_json[child_key2].append([act_currency, data_json[parent_field][i][1]])
                        
                    # data_json[child_key1].append(data_json[parent_field][i][1])
                else:
                    if len(act_currency)>0:
                        if prediction_flag:
                            data_json[child_key2]=[[act_currency, data_json[parent_field][i][1], data_json[parent_field][i][2]]]

                        else:
                            data_json[child_key2]=[[act_currency, data_json[parent_field][i][1]]]
                            
    if child_key2 in pred_actual_list:
        index_to_delete = []
        for i in range(len(data_json[child_key2])):
            act_currency, act_amount = extract_currency_and_amount(str(data_json[child_key2][i][0]))
            if len(act_amount)>0 and child_key1 in pred_actual_list:
                if prediction_flag:
                    data_json[child_key1].append([act_amount, data_json[child_key2][i][1], data_json[child_key2][i][2]])
                else:
                    data_json[child_key1].append([act_amount, data_json[child_key2][i][1]])

                # data_json[child_key1].append(data_json[parent_field][i][1])
            else:
                if len(act_amount)>0:
                    if prediction_flag:
                        data_json[child_key1]=[[act_amount, data_json[child_key2][i][1], data_json[child_key2][i][2]]]

                    else:
                        data_json[child_key1]=[[act_amount, data_json[child_key2][i][1]]]

  
            if len(act_currency)>0: #and child_key2 in pred_actual_list:
                data_json[child_key2][i][0] = act_currency   



    if child_key1 in pred_actual_list:
        for i in range(len(data_json[child_key1])):
            act_currency, act_amount = extract_currency_and_amount(str(data_json[child_key1][i][0]))
            if len(act_currency)>0 and child_key2 in pred_actual_list:
                if prediction_flag:
                    data_json[child_key2].append([act_currency, data_json[child_key1][i][1], data_json[child_key1][i][2]])
                else:
                    data_json[child_key2].append([act_currency, data_json[child_key1][i][1]])
                # data_json[child_key1].append(data_json[parent_field][i][1])
            else:
                if len(act_currency)>0:
                    if prediction_flag:
                        data_json[child_key2]=[[act_currency, data_json[child_key1][i][1], data_json[child_key1][i][2]]]

                    else:
                        data_json[child_key2]=[[act_currency, data_json[child_key1][i][1]]]


            if len(act_amount)>0: #and child_key2 in pred_actual_list:
                data_json[child_key1][i][0] = act_amount
     
    



def amount_currency_overlapping(key1, key2, labels, predicted):
    
    actual_list = list(labels)
    pred_list = list(predicted)
    if key1 in actual_list:
            # exit('???????????????????????????????????')
        index_to_delete = []
        for i in range(len(labels[key1])):
            amount_list = list()
            sum_insured = labels[key1][i][0].split()
            for j in sum_insured:
                act_currency, act_amount = extract_currency_and_amount(str(j))
                if len(act_currency)>0 and key2 in actual_list:
                    labels[key2].append([act_currency, labels[key1][i][1]])
                    # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                else:
                    if len(act_currency)>0:
                        labels[key2]=[[act_currency, labels[key1][i][1]]]
                if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                    amount_list.append(act_amount)
            if len(amount_list)>0:
                labels[key1][i][0] = amount_list
            else:
                index_to_delete.append(i)
        if len(index_to_delete)>0:
            for d in index_to_delete:
                del labels[key1][d]
            if len(labels[key1])==0:
                del labels[key1]
                # del labels[key1][i]
            #else => needed to delete the i th element in the sum_insured_amount , if required.
    # if key1 in actual_list:

    if key1 in pred_list:
        index_to_delete = []
        for i in range(len(predicted[key1])):
            amount_list = []
            sum_insured = predicted[key1][i][0].split()
            for j in sum_insured:
                act_currency, act_amount = extract_currency_and_amount(str(j))
                if len(act_currency)>0 and key2 in pred_list:
                    predicted[key2].append([act_currency, predicted[key1][i][1], predicted[key1][i][2]])
                    # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                else:
                    if len(act_currency)>0:
                        predicted[key2]=[[act_currency, predicted[key1][i][1], predicted[key1][i][2]]]
    
                
                if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                    amount_list.append(act_amount)
            if len(amount_list)>0:
                predicted[key1][i][0] = amount_list
            else:
                index_to_delete.append(i)
        if len(index_to_delete)>0:
            for d in index_to_delete:
                del predicted[key1][d]
            if len(predicted[key1])==0:
                del predicted[key1]


    if key2 in actual_list:
        index_to_delete = []
        for i in range(len(labels[key2])):
            if file[0:-11]=='Insurance_Certificate_34_page_0':
                print(labels[key2])
                # exit()
            currency_list = []
            sum_insured = labels[key2][i][0].split()
            if file[0:-11]=='Insurance_Certificate_34_page_0':
                print(labels[key2])
                print(sum_insured)
                # exit()
            for j in sum_insured:
                act_currency, act_amount = extract_currency_and_amount(str(j))
                if len(act_amount)>0 and key1 in actual_list:
                    labels[key1].append([act_amount, labels[key2][i][1]])
                    # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                else:
                    if len(act_amount)>0:
                        labels[key1]=[[act_amount, labels[key2][i][1]]]
    
                
                if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                    currency_list.append(act_currency)
                    
                                                                                                                                                                                                                                                                                                        
            if len(currency_list)>0:
                labels[key2][i][0] = currency_list

            else:
                index_to_delete.append(i)
        if len(index_to_delete)>0:
            for d in index_to_delete:
                del labels[key2][d]
            if len(labels[key2])==0:
                del labels[key2]
                
    if key2 in pred_list:
        index_to_delete = []
        for i in range(len(predicted[key2])):
            currency_list = []
            sum_insured = predicted[key2][i][0].split()
            for j in sum_insured:
                act_currency, act_amount = extract_currency_and_amount(str(j))
                if len(act_amount)>0 and key1 in pred_list:
                    predicted[key1].append([act_amount, predicted[key2][i][1], predicted[key2][i][2]])
                    # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                else:
                    if len(act_amount)>0:
                        predicted[key1]=[[act_amount, predicted[key2][i][1], predicted[key2][i][2]]]
    
                
                if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                    currency_list.append(act_currency)
                
            if len(currency_list)>0:
                predicted[key2][i][0] = currency_list
            else:
                index_to_delete.append(i)
        if len(index_to_delete)>0:
            for d in index_to_delete:
                del predicted[key2][d]
            if len(predicted[key2])==0:
                del predicted[key2]