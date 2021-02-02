import torch
import transformers
from utils import *


class T5mini(torch.nn.Module):
    def __init__(self, path, teacher_size=None, small=True, tiny=True, device="cuda:0"):
        super().__init__()
        self.tiny = tiny
        self.device = device
        if small:
            self.T5mini = transformers.T5ForConditionalGeneration.from_pretrained(
                path
            )
        else:
            self.T5mini = transformers.T5ForConditionalGeneration(
                transformers.T5Config(d_ff=1024, d_kv=64, d_model=512, decoder_start_token_id=0, dropout_rate=0.1,
                                      eos_token_id=1,
                                      feed_forward_proj="gated-gelu", initializer_factor=1.0, is_encoder_decoder=True,
                                      layer_norm_epsilon=1e-06, num_decoder_layers=6, num_heads=6, num_layers=6,
                                      output_past=True,
                                      pad_token_id=0, relative_attention_num_buckets=32, tie_word_embeddings=False,
                                      vocab_size=32128))
            self.T5mini.load_state_dict(torch.load(path))
        if teacher_size is not None:
            self.W = torch.nn.Linear(self.T5mini.shared.embedding_dim, teacher_size, bias=False)
        if self.tiny:
            self.mapping = {0: 0, 1: 2, 2: 4, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12}
            self.attention_mapping = dict(zip(list(range(0, 6)), list(range(0, 12, 2))))
            self.attention_mapping[5] = 11
            self.decoder_hidden_W = [torch.nn.Linear(self.T5mini.shared.embedding_dim, teacher_size, bias=False).to(self.device)
                                     for _ in range(9)]

            self.encoder_hidden_W = [torch.nn.Linear(self.T5mini.shared.embedding_dim, teacher_size, bias=False).to(self.device)
                                     for _ in range(9)]
            self.mse = torch.nn.MSELoss()

    def forward(self, input_ids, labels, attention_mask=None, teacher=None):

        if teacher is None:
            return self.T5mini(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        else:
            student_outputs = self.T5mini(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                          output_hidden_states=True, use_cache=False, output_attentions=True)
            distill_loss = kl_div(student_outputs[1], teacher[1])
            if self.tiny:
                loss = 0
                teacher_loss, teacher_logits, teacher_decoder_hidden_states, teacher_decoder_attentions, teacher_cross_attentions, \
                teacher_encoder_last_hidden_state, teacher_encoder_hidden_states, teacher_encoder_attentions = teacher[0], teacher[1],\
                                                                                                               teacher[2], teacher[3],\
                                                                                                               teacher[4], teacher[5],\
                                                                                                               teacher[6], teacher[7]
                student_loss, student_logits, student_decoder_hidden_states, student_decoder_attentions, student_cross_attentions, \
                student_encoder_last_hidden_state, student_encoder_hidden_states, student_encoder_attentions = student_outputs[0], student_outputs[1],\
                                                                                                               student_outputs[2], student_outputs[3],\
                                                                                                               student_outputs[4], student_outputs[5],\
                                                                                                     student_outputs[6], student_outputs[7]

                # decoder hidden loss
                count = 0
                new_teacher_dec_hid = [teacher_decoder_hidden_states[self.mapping[key]] for key in
                                       self.mapping]
                new_student_dec_hid = [self.decoder_hidden_W[i](student_decoder_hidden_states[i].to(self.device))
                                       for i in range(len(student_decoder_hidden_states))]
                for student_hid, teacher_hid in zip(new_student_dec_hid, new_teacher_dec_hid):
                    tmp_loss = self.mse(student_hid, teacher_hid)
                    count += 1
                    loss += tmp_loss

                # encoder hidden loss
                new_teacher_enc_hid = [teacher_encoder_hidden_states[self.mapping[key]] for key in
                                       self.mapping]
                new_student_enc_hid = [self.encoder_hidden_W[i](student_encoder_hidden_states[i])
                                       for i in range(len(student_encoder_hidden_states))]
                for student_hid, teacher_hid in zip(new_student_enc_hid, new_teacher_enc_hid):
                    tmp_loss = self.mse(student_hid, teacher_hid)
                    count += 1
                    loss += tmp_loss

                # decoder attention loss
                new_teacher_dec_atts = [teacher_decoder_attentions[self.mapping[key]] for key in
                                        range(len(student_decoder_attentions))]
                for student_att, teacher_att in zip(student_decoder_attentions, new_teacher_dec_atts):
                    new_teacher_att = torch.zeros(student_att.shape).to(self.device)
                    for i in range(student_att.shape[1]):
                        new_teacher_att[:, i, :, :] = teacher_att[:, self.attention_mapping[i], :, :]

                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                              student_att)
                    new_teacher_att = torch.where(new_teacher_att <= -1e2, torch.zeros_like(new_teacher_att).to(self.device),
                                              new_teacher_att)

                    tmp_loss = self.mse(student_att, new_teacher_att)
                    count += 1
                    loss += tmp_loss

                # encoder attention loss
                new_teacher_enc_atts = [teacher_encoder_attentions[self.mapping[key]] for key in
                                        range(len(student_encoder_attentions))]
                for student_att, teacher_att in zip(student_encoder_attentions, new_teacher_enc_atts):
                    new_teacher_att = torch.zeros(student_att.shape).to(self.device)
                    for i in range(student_att.shape[1]):
                        new_teacher_att[:, i, :, :] = teacher_att[:, self.attention_mapping[i], :, :]

                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                              student_att)
                    new_teacher_att = torch.where(new_teacher_att <= -1e2,
                                                  torch.zeros_like(new_teacher_att).to(self.device),
                                                  new_teacher_att)

                    tmp_loss = self.mse(student_att, new_teacher_att)
                    count += 1
                    loss += tmp_loss



                loss += distill_loss
                loss += student_outputs[0]
                count += 2
                loss /= count
                emb_loss = tmp_loss
            else:
                emb_loss = cos_loss(self.W(student_outputs[3]), teacher[3])
                loss = (student_outputs[0] + distill_loss +
                        emb_loss) / 3.0
            return student_outputs[0], loss, emb_loss, distill_loss
