import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, PreTrainedModel

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

USE_CUDA = False
device = get_device()
if device.type == 'cuda':
    USE_CUDA = True

bert_path = 'indobenchmark/indobert-base-p2'
HIDDEN_DIM = 768
OUTPUT_DIM = 2 # 2 if Binary Classification
N_LAYERS = 1 # 2
BIDIRECTIONAL = True
DROPOUT = 0.2 # 0.2

class IndoBERTBiLSTM(PreTrainedModel):
    config_class = BertConfig
    def __init__(self, bert_config):
        super().__init__(bert_config)
        self.output_dim = OUTPUT_DIM
        self.n_layers = 1
        self.hidden_dim = HIDDEN_DIM
        self.bidirectional = BIDIRECTIONAL

        self.bert = BertModel.from_pretrained(bert_path)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.output_layer = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim, self.output_dim)

    def forward(self, input_ids, attention_mask):

        hidden = self.init_hidden(input_ids.shape[0])
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state

        lstm_output, (hidden_last, cn_last) = self.lstm(sequence_output, hidden)

        hidden_last_L=hidden_last[-2]
        hidden_last_R=hidden_last[-1]
        hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1) #[16, 1536]

        # apply dropout
        out = self.dropout(hidden_last_out)

        # output layer
        logits = self.output_layer(out)

        return logits

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        if (USE_CUDA):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )

        return hidden