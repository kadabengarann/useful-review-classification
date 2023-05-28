import torch
import re
import streamlit as st
from transformers import BertTokenizer, BertModel
from model import IndoBERTBiLSTM, IndoBERTModel

# Config
MAX_SEQ_LEN = 128
bert_path = 'indolem/indobert-base-uncased'
MODELS_PATH = ["kadabengaran/IndoBERT-Useful-App-Review",
               "kadabengaran/IndoBERT-BiLSTM-Useful-App-Review"]

            #    "kadabengaran/IndoBERT-BiLSTM-Useful-App-Review"]
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
    # menghapus karakter khusus kecuali tanda baca seperti titik, koma, dan tanda tanya
    # text = re.sub(r"[^a-zA-Z0-9.,!?]+", " ", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # text = re.sub(r"'\s+|\s+'", " ", text)  # replace apostrophe with space if it's surrounded by whitespace
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
    model1 = IndoBERTBiLSTM.from_pretrained(MODELS_PATH[0],
                                     bert,
                                     HIDDEN_DIM,
                                     OUTPUT_DIM,
                                     N_LAYERS, BIDIRECTIONAL,
                                     DROPOUT)
    model2 = IndoBERTModel.from_pretrained(MODELS_PATH[1],
                                     bert,
                                     OUTPUT_DIM)
    return model1, model2


def predict(text, model, tokenizer, device):
    
    # model = torch.load(model_path, map_location=device)
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

def main():
    """App Review Classifier"""
    # st.title("Klasifikasi Ulasan APlikasi")
    # st.subheader("ML App with Streamlit")
    html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Klasifikasi Ulasan Aplikasi yang Berguna</h1>
	</div>

	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    # st.info("Prediction with ML")

    input_text = st.text_area("Enter Text Here", placeholder="Type Here")
    all_ml_models = ["IndoBERT", "IndoBERT-BiLSTM"]
    model_choice = st.selectbox("Select Model", all_ml_models)

    tokenizer = load_tokenizer(bert_path)
    device = get_device()
    model1, model2 = load_model()
    
    prediction = 0
    prediction_labels = {'Not Useful': 0, 'Useful': 1}
    if st.button("Classify"):
        st.text("Original Text:\n{}".format(input_text))
        if model_choice == 'IndoBERT':
            prediction = predict(input_text, model1, tokenizer, device)
        elif model_choice == 'IndoBERT-BiLSTM':
            prediction = predict(input_text, model2, tokenizer, device)
        final_result = get_key(prediction, prediction_labels)
        st.success("Review Categorized as:: {}".format(final_result))
    # st.sidebar.subheader("About")

if __name__ == '__main__':
    main()
