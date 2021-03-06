import os
import json

import torch
from torch import nn
import numpy

from toolkit.utils import WORD_MAP_FILENAME, LSTMCell, load_pretrained_embedding_from_file
from toolkit.models.captioning_model_context import (
        CaptioningEncoderDecoderContextModel, CaptioningDecoder
)

class BUTDContextLSTMModel(CaptioningEncoderDecoderContextModel):
    def __init__(self, args, device):
        super(BUTDContextLSTMModel, self).__init__()

        # Read word map
        word_map_filename = os.path.join(args.dataset_splits_dir, WORD_MAP_FILENAME)
        with open(word_map_filename) as f:
            word_map = json.load(f)

        # Pre-trained Embeddings
        if args.embeddings_path:
            embeddings, embed_dim = load_pretrained_embedding_from_file(args.embeddings_path, word_map)
        else:
            embeddings, embed_dim = None, args.embeddings_dim

        self.decoder = TopDownDecoder(
            word_map=word_map,
            embed_dim=embed_dim,
            encoder_output_dim=args.encoder_output_dim,
            pretrained_embeddings=embeddings,
            embeddings_freeze=args.embeddings_freeze,
            language_lstm_dim=args.language_lstm_dim,
            attention_lstm_dim=args.attention_lstm_dim,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
            device=device
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser"""
        group = parser.add_argument_group("Bottom-Up Top-Down Context")
        group.add_argument("--encoder-output-dim", type=int, default=2048)
        group.add_argument("--embeddings-dim", type=int, default=1000)
        group.add_argument("--attention-dim", type=int, default=512)
        group.add_argument("--attention-lstm-dim", type=int, default=1000)
        group.add_argument("--language-lstm-dim", type=int, default=1000)
        group.add_argument("--teacher-forcing", type=float, default=1)
        group.add_argument("--dropout", type=float, default=0.0)
        group.add_argument("--embeddings-freeze", type=bool, default=False)
        group.add_argument("--decoder-learning-rate", type=float, default=1e-4)

        return group


class TopDownDecoder(CaptioningDecoder):
    def __init__(self, word_map, embed_dim=1000, encoder_output_dim=2048,
                 pretrained_embeddings=None, embeddings_freeze=False,
                 language_lstm_dim=1000, attention_lstm_dim=1000, 
                 attention_dim=512, dropout=0.0, device=None):
        super(TopDownDecoder, self).__init__(word_map, embed_dim, encoder_output_dim,
                                             pretrained_embeddings, embeddings_freeze)
                
        self.attention_dim = attention_dim
        self.language_lstm_size = language_lstm_dim
        self.attention_lstm_size = attention_lstm_dim

        # LSTM layers
        self.attention_lstm = AttentionLSTM(self.embed_dim, language_lstm_dim, 
                                            encoder_output_dim, attention_lstm_dim)
        self.language_lstm = LanguageLSTM(attention_lstm_dim, encoder_output_dim, language_lstm_dim)

        # Attention Layer
        self.attention = VisualAttention(encoder_output_dim, attention_lstm_dim, attention_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(language_lstm_dim, self.vocab_size, bias=True)
        # self.fc_w = nn.Linear(language_lstm_dim, self.vocab_size, bias=True)
        # self.fc_t = nn.Linear(language_lstm_dim, self.vocab_size, bias=True)

        # Linear layers to find initial states of LSTMs
        self.init_h1 = nn.Linear(encoder_output_dim, self.attention_lstm.lstm_cell.hidden_size)
        self.init_c1 = nn.Linear(encoder_output_dim, self.attention_lstm.lstm_cell.hidden_size)

        self.init_h2 = nn.Linear(encoder_output_dim, self.language_lstm.lstm_cell.hidden_size)
        self.init_c2 = nn.Linear(encoder_output_dim, self.language_lstm.lstm_cell.hidden_size)

        self.softmax = nn.Softmax()
        self.rev_word_map = {v: k for k, v in word_map.items()}

        self.device=device

        #loss = nn.NLLLoss()

    def init_hidden_states(self, encoder_out, context_retrieval=None):
        v_mean = encoder_out.mean(dim=1)
        h1 = self.init_h1(v_mean)
        c1 = self.init_c1(v_mean)
        h2 = self.init_h2(v_mean)
        c2 = self.init_c2(v_mean)
        states = [h1, c1, h2, c2]

        context_model=context_retrieval.sentence_model
        cont_states = context_model.init_hidden_states(encoder_out)
        context_model.states= cont_states

        return states

    def forward_step(self, encoder_output, prev_word_embeddings, states):
        """Perform a single decoding step."""

        v_mean = encoder_output.mean(dim=1)
        h1, c1, h2, c2 = states

        h1, c1 = self.attention_lstm(h1, c1, h2, v_mean, prev_word_embeddings)
        v_hat = self.attention(encoder_output, h1)
        h2, c2 = self.language_lstm(h2, c2, h1, v_hat)
        scores = self.fc(self.dropout(h2))

        states = [h1, c1, h2, c2]

        return scores, states, None

    def interpolate(self, scores, encoder_output, prev_words, retrieval, target_lookup, interpolation=0.25, k_neighbours=16):
        #print("socres", scores)
        softmax_scores = self.softmax(scores)
        #print("socres log softmax", softmax_scores)

        context_model=retrieval.sentence_model
        scores, states = context_model.forward_step(encoder_output, prev_words, context_model.states)
        _, _, h2, _ = states
        
        if self.training:
            nearest_targets, distances= retrieval.retrieve_nearest_for_train_query(h2, k_neighbours)
        else:
            nearest_targets, distances= retrieval.retrieve_nearest_for_val_or_test_query(h2,k_neighbours)

        scores_size=scores.size()
        softmax_nearest = torch.zeros(scores_size[0], k_neighbours,scores_size[1]).to(self.device)
        nearest_probs = self.softmax(-1.*torch.tensor(distances)).to(self.device)
        ind=torch.arange(0, k_neighbours).expand(softmax_nearest.size(0), -1).to(self.device)
        ind_batch=torch.arange(0, scores_size[0]).reshape(-1,1).to(self.device)

        softmax_nearest[ind_batch, ind,nearest_targets] = nearest_probs
        
        softmax_nearest = softmax_nearest.sum(1)
     
        softmax_interpolation=interpolation*softmax_nearest.to(self.device) + (1-interpolation)*softmax_scores

        softmax_interpolation = torch.clamp(softmax_interpolation, 1e-9, 1 - 1e-9)
        return torch.log(softmax_interpolation)

class AttentionLSTM(nn.Module):
    def __init__(self, embed_dim, lang_lstm_dim, encoder_dim, hidden_size):
        super(AttentionLSTM, self).__init__()
        self.lstm_cell = LSTMCell(
            input_size=lang_lstm_dim + encoder_dim + embed_dim, 
            hidden_size=hidden_size,
            bias=True
        )

    def forward(self, h1, c1, h2, v_mean, prev_word_embeddings):
        input_feats = torch.cat((h2, v_mean, prev_word_embeddings), dim=1)
        h_out, c_out = self.lstm_cell(input_feats, (h1, c1))
        return h_out, c_out


class LanguageLSTM(nn.Module):
    def __init__(self, attn_lstm_dim, viz_attn_dim, hidden_size):
        super(LanguageLSTM, self).__init__()
        self.lstm_cell = LSTMCell(
            input_size=attn_lstm_dim + viz_attn_dim, 
            hidden_size=hidden_size, 
            bias=True
        )

    def forward(self, h2, c2, h1, v_hat):
        input_feats = torch.cat((h1, v_hat), dim=1)
        h_out, c_out = self.lstm_cell(input_feats, (h2, c2))
        return h_out, c_out


class VisualAttention(nn.Module):
    def __init__(self, encoder_out, attn_lstm_dim, hidden_size):
        super(VisualAttention, self).__init__()
        self.linear_image_features = nn.Linear(encoder_out, hidden_size, bias=False)
        self.linear_attn_lstm = nn.Linear(attn_lstm_dim, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_attention = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images, h1):
        embedded_image_features = self.linear_image_features(images)
        embedded_attn_lstm = self.linear_attn_lstm(h1).unsqueeze(1)

        all_feats_emb = embedded_image_features + embedded_attn_lstm.repeat(1, images.size()[1], 1)

        activate_feats = self.tanh(all_feats_emb)
        attention = self.linear_attention(activate_feats)
        normalized_attention = self.softmax(attention)

        weighted_feats = normalized_attention * images
        attn_weighted_image_features = weighted_feats.sum(dim=1)
        return attn_weighted_image_features