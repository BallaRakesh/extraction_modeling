import re


def clean_text_sep(text, index1=None):
    # Remove tokens from the end of text only
    # Using rstrip() to handle any trailing spaces
    if index1 != None:
        print('came here...')
        text[index1] = re.sub(r'\s*\[SEP\]$|\s*SEP$', '', text[index1].rstrip())
    else:
        text = re.sub(r'\s*\[SEP\]$|\s*SEP$', '', text.rstrip())
    return text

predicted_label = ['# inv01206371', [1955, 300, 2386, 349]]
predicted_label = clean_text_sep(predicted_label,  index1=0)

print(predicted_label)
exit('OK')












def clean_text(text):
    # Remove tokens from the end of text only
    # Using rstrip() to handle any trailing spaces
    text = re.sub(r'\s*\[SEP\]$|\s*SEP$', '', text.rstrip())
    return text

# Test cases
texts = [
    "3 / 4 [SEP]",             # Token at end
    "290 . 00 SEP",            # Token at end
    "[SEP] start text",        # Token at start (shouldn't be removed)
    "middle SEP text",         # Token in middle (shouldn't be removed)
    "no token text",           # No token
    "multiple SEP [SEP]",      # Multiple tokens, only end should be removed
    "text[SEP]",               # No space before token
    "text SEP"                 # With space before token
]

for text in texts:
    cleaned = clean_text(text)
    print(f"Original: '{text}'")
    print(f"Cleaned : '{cleaned}'")
    print("-" * 40)
    
    
exit('OK')




from fuzzywuzzy import fuzz

str1 = 'abc     def xyq'
str2 = 'abcdef     xyq'

# Remove spaces
str1_no_spaces = str1.replace(' ', '')
str2_no_spaces = str2.replace(' ', '')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>',str1)
print()
# Now compare
ratio = fuzz.ratio(str1_no_spaces, str2_no_spaces)

print(f"Original strings: '{str1}' vs '{str2}'")
print(f"Without spaces: '{str1_no_spaces}' vs '{str2_no_spaces}'")
print(f"Ratio: {ratio}")