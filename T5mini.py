import torch
import transformers
from utils import *


class T5mini(torch.nn.Module):
    def __init__(self, path, teacher_size=None, small=True):
        super().__init__()
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

    def forward(self, input_ids, labels, attention_mask=None, teacher=None):
        if teacher is None:
            return self.T5mini(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        else:
            student_outputs = self.T5mini(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            distill_loss = kl_div(student_outputs[1], teacher[1])
            emb_loss = cos_loss(self.W(student_outputs[3]), teacher[3])
            loss = (student_outputs[0] + distill_loss +
                    emb_loss) / 3.0
            return student_outputs[0], loss, emb_loss,  distill_loss
