#!/bin/bash
conda activate geolayoutlm
echo "$(which python)"
cd /New_Volume/trade-finance/final_delivery/code/Trade-Finance

# Arrays to store different types of elements
declare -a prod_names=("bills" "lc" "bg")
declare -a doc_bills=("ci" "ic" "coo" "pl" "bol" "awb" "boe")
declare -a doc_lc=(po, pi)
declare -a doc_bg=(bgc)

# Get the length of the arrays
array_length_prod_names=${#prod_names[@]}
array_length_doc_bills=${#doc_bills[@]}
array_length_doc_lc=${#doc_lc[@]}
array_length_doc_bg=${#doc_bg[@]}


# printing the length of the arrays
echo "Number of products: $array_length_prod_names, Number of documentsin bills: $array_length_doc_bills,\
Number of documents in lc: $array_length_doc_lc, Number of documents in bg: $array_length_doc_bg"

# Iterate over the arrays
for ((i=0; i<$array_length_doc_bills; i++)); do
  name="${doc_bills[$i]}"
  # Print information about each element
  echo "Document Name: $name"
  /home/admin1/anaconda3/envs/geolayoutlm/bin/python -m src.main.extraction.master_data_merger_latest
done
