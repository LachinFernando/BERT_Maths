import streamlit as st
import requests
import json
from transformers import AutoTokenizer

###################################
#Needful functions

#batch encode function
def batch_encode(tokenizer, texts, batch_size=16, max_length=162):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained transformer model.
    
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""
    
    input_ids = []
    attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             padding='max_length', #implements dynamic padding
                                             truncation=True,
                                             max_length = max_length,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    print(len(input_ids))
    
    return input_ids, attention_mask

#disabling hashing function
#load the tokenizer
#@st.cache(hash_funcs={"MyUnhashableClass": lambda _: None})
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('tokenizer')

#loading the tokenizer
final_tokenizer = load_tokenizer()

def get_prediction(data):
  url = 'https://aut18k48yh.execute-api.us-east-1.amazonaws.com/math_functions/math_api'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  #print(response)
  return response

#formulas notedown
def math_func(*args):
    for function in args:
        st.markdown(f"- **{function}**")

#####################################
#WEB APP


#you can select any suitable title here
st.title("Math Solver")

#image
st.image("https://img.freepik.com/premium-psd/mathematics-geometry-formulas-with-green-pencils_23-2148347756.jpg?w=2000")

#primarily can go on with 3 tabs
tab1, tab2, tab3 = st.tabs(["About the Project 📜", "Instructions 📝", "Let's Solve Math 🔢"])

with tab1:
    #subheader to display about the project
    st.header("About the Website")

    #write about the website
    st.markdown("- I have used python....................")

with tab2:
    #subheader as the instructions
    st.header("Instructions on how to use the Website")

    #steps
    st.markdown("- You can your math question....................")

with tab3:
    #subheading
    st.header("Get a help to solve your MATH problem")

    #another subheading
    st.subheader("Please enter your math problem")

    #text input field
    problem = st.text_input("Enter your problem")

    if not problem:
        st.error("Please Enter the Problme to Proceed with", icon="🚨")
    else:
        sentence_ids, sentence_attentions = batch_encode(final_tokenizer, [problem])

        #setting parameters
        event = {
            "httpmethod": "POST",
            "body": {
            "sentence": sentence_ids,
            "attention": sentence_attentions
            }
        }

        #getting the response from the API
        response = get_prediction(data = event)
        response = json.loads(response)
        helpful_functions = response['body']['predictions']
        
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Problem You Entered")
            st.markdown(f"**{problem}**")
        with col2:
            st.subheader("Formulas You Will Need when Solving..")
            math_func(*helpful_functions)