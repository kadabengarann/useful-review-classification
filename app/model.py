import torch.nn as nn
from transformers import PreTrainedModel, BertConfig

USE_CUDA = False

class IndoBERTBiLSTM(PreTrainedModel):
    config_class = BertConfig
    def __init__(self, bert_config, bert_pretrained_path, hidden_dim, num_classes, n_layers, bidirectional, dropout):
        super().__init__(bert_config)
        self.output_dim = num_classes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.bert = bert_pretrained_path
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.hidden_layer = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim * 2 if bidirectional else hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):

        hidden = self.init_hidden(input_ids.shape[0])
        # print("hidden : ", type(hidden))
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = output.last_hidden_state

        # apply dropout
        sequence_output = self.dropout(sequence_output)
        # print('output size of the bert:', last_hidden_state.size())
    
        lstm_output, (hidden_last, cn_last) = self.lstm(sequence_output, hidden)
        # print('output size of the LSTM:', lstm_output.size())
        lstm_output = self.dropout(lstm_output)

        # global pooling
        lstm_output = lstm_output.permute(0, 2, 1)
        pooled_output = self.global_pooling(lstm_output).squeeze()

        # pass through hidden layer
        hidden_layer_output = self.hidden_layer(pooled_output)
        hidden_layer_output = self.relu(hidden_layer_output)

        # output layer
        logits = self.output_layer(hidden_layer_output)
        # logits = nn.Softmax(dim=1)(logits)

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
    

class IndoBERTModel(PreTrainedModel):
    config_class = BertConfig
    def __init__(self, bert_config, bert_pretrained, num_classes):
        super().__init__(bert_config)
        self.bert = bert_pretrained
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits