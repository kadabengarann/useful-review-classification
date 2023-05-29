try:
    import torch
 
    import pandas as pd
    import streamlit as st
    import re
    import streamlit as st
    from transformers import BertTokenizer, BertModel
    from model import IndoBERTBiLSTM, IndoBERTModel
except Exception as e:
    print(e)
 
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
# Config
MAX_SEQ_LEN = 128
# bert_path = './local/base-indobert'
bert_path = 'indolem/indobert-base-uncased'
MODELS_PATH = ["kadabengaran/IndoBERT-Useful-App-Review",
               "kadabengaran/IndoBERT-BiLSTM-Useful-App-Review"]
# MODELS_PATH = ["./local/indobert1",
#                "./local/indobert2"]

MODELS_NAME = ["IndoBERT-BiLSTM", "IndoBERT"]
LABELS = {'Not Useful': 0, 'Useful': 1}
            
HIDDEN_DIM = 768
OUTPUT_DIM = 2 # 2 if Binary
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2

# Get the Keys
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return tokenizer


def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r"\s+", " ", text)  # replace multiple whitespace characters with a single space
    text = re.sub(r'[0-9]', ' ', text) #remove number
    text = text.lower()
    return text


def preprocess(text, tokenizer, max_seq=MAX_SEQ_LEN):
    return tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq,
                                 pad_to_max_length=True,
                                 return_attention_mask=True,
                                 return_tensors='pt'
                                 )
    
def load_model():
    bert = BertModel.from_pretrained(bert_path)
    
	# Load the model
    model_combined = IndoBERTBiLSTM.from_pretrained(MODELS_PATH[0],
                                     bert,
                                     HIDDEN_DIM,
                                     OUTPUT_DIM,
                                     N_LAYERS, BIDIRECTIONAL,
                                     DROPOUT)
    model_base = IndoBERTModel.from_pretrained(MODELS_PATH[1],
                                     bert,
                                     OUTPUT_DIM)
    return model_combined, model_base

def predict_single(text, model, tokenizer, device):
    
    if device.type == 'cuda':
        model.cuda()
        
    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply preprocessing to the new sentence
    new_sentence = remove_special_characters(text)
    encoding = preprocess(new_sentence, tokenizer)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim=0)
    test_attention_mask = torch.cat(test_attention_mask, dim=0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        outputs = model(test_ids.to(device),
                        test_attention_mask.to(device))
    print("output ", outputs)
    predictions = torch.argmax(outputs, dim=-1)
    print("output ", predictions)
    return predictions.item()

def predict_multiple(data, model, tokenizer, device):
    input_ids = []
    attention_masks = []
    for row in data.tolist():
        # Apply remove_special_characters function to title column
        text = remove_special_characters(row)
        text = preprocess(text, tokenizer)
        input_ids.append(text['input_ids'])
        attention_masks.append(text['attention_mask'])
        
    predictions = []
    
    with torch.no_grad():
        for i in range(len(input_ids)):
            test_ids = input_ids[i]
            test_attention_mask = attention_masks[i]
            outputs = model(test_ids.to(device), test_attention_mask.to(device))
            prediction = torch.argmax(outputs, dim= -1)
            prediction_label = get_key(prediction.item(), LABELS)
            predictions.append(prediction_label)
            
    return predictions

tab_labels = ["Single Input", "Multiple Input"]
class App:
    
    print("Loading All")
    def __init__(self):
        self.fileTypes = ["csv"]
        self.default_tab_selected = tab_labels[0]
        self.input_text = None
        self.input_file = None
        
    def run(self):
        self.init_session_state()  # Initialize session state
        tokenizer = load_tokenizer(bert_path)
        device = get_device()
        model_combined, model_base = load_model()
        """App Review Classifier"""
        html_temp = """
        <div style="background-color:blue;padding:10px">
        <h1 style="color:white;text-align:center;">Klasifikasi Ulasan Aplikasi yang Berguna</h1>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        self.render_tabs()
        st.divider()
        model_choice = self.render_model_selection()
        if model_choice:
            if model_choice == MODELS_NAME[0]:
                model = model_combined
            elif model_choice == MODELS_NAME[1]:
                model = model_base
            self.render_process_button(model, tokenizer, device)

    def init_session_state(self):
        if "tab_selected" not in st.session_state:
            st.session_state.tab_selected = tab_labels[0]

    def render_model_selection(self):
        model_choice = st.selectbox("Select Model", MODELS_NAME)
        return model_choice
    
    def render_tabs(self):
        tab_selected = st.session_state.get('tab_selected', self.default_tab_selected)
        tab_selected = st.sidebar.radio("Select Input Type", tab_labels)
        # tab1, tab2 = st.tabs(tab_labels)

        if tab_selected == tab_labels[0]:
            self.render_single_input()
        elif tab_selected == tab_labels[1]:
            self.render_multiple_input()
            
        st.session_state.tab_selected = tab_selected

    def render_single_input(self):
        self.input_text = st.text_area("Enter Text Here", placeholder="Type Here")

    def render_multiple_input(self):
        """
        Upload File
        """
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)


        if not file:
            st.info("Please upload a file of type: " + ", ".join(self.fileTypes))
            return

        data = pd.read_csv(file)
        
        placeholder = st.empty()
        placeholder.dataframe(data.head(10))


        header_list = data.columns.tolist()
        header_list.insert(0, "---------- select column -------------")
        ques = st.radio("Select column to process", header_list, index=0)

        if header_list.index(ques) == 0:
            st.warning("Please select a column to process")
            return

        df_process = data[ques]
        self.input_file = data
        self.process_file = df_process
        
    def render_process_button(self, model, tokenizer, device):
        if st.button("Process"):
            if st.session_state.tab_selected == tab_labels[0]:
                input_text = self.input_text
                if input_text:
                    prediction = predict_single(input_text, model, tokenizer, device)
                    prediction_label = get_key(prediction, LABELS)
                    st.write("Prediction:", prediction_label)
            elif st.session_state.tab_selected == tab_labels[1]:
                df_process = self.process_file
                if df_process is not None:
                    prediction = predict_multiple(df_process, model, tokenizer, device)
                    
                    st.divider()
                    st.write("Classification Result")
                    input_file = self.input_file
                    input_file["classification_result"] = prediction
                    st.dataframe(input_file.head(10))
                    st.download_button(
                        label="Download Result",
                        data=input_file.to_csv().encode("utf-8"),
                        file_name="classification_result.csv",
                        mime="text/csv",
                    )
    

if __name__ == "__main__":
    app = App()
    app.run()