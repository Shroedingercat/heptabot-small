from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.bleu_score import sentence_bleu
import torch
import torch.nn.functional as F


from transformers import T5TokenizerFast, T5Model, T5Config
tokenizer = T5TokenizerFast.from_pretrained('t5-small')


def get_corpus_gleu_jfleg(pred, target, max_length=512):
    bleu = 0
    for i in range(pred.shape[0]):
        p = (tokenizer.decode(pred[i], skip_special_tokens=True)).split(" ")
        t = [(tokenizer.decode(target[i][j][0], skip_special_tokens=True)).split(" ") for j in range(len(target[i]))]
        bleu += sentence_gleu(t, p)
    return p, t, bleu/pred.shape[0]


def clean_repeats(s):
    return s[:s[64:].find(s[:64])+63]


def get_bleu(pred, target, max_length=512):
    bleu = 0
    for i in range(pred.shape[0]):
        p = (tokenizer.decode(pred[i], skip_special_tokens=True)).split(" ")
        t = (tokenizer.decode(target[i], skip_special_tokens=True)).split(" ")
        bleu += sentence_gleu([t], p)
    return p, t, bleu / pred.shape[0]


def get_gleu(pred, target, max_length=512):
    bleu = 0
    for i in range(pred.shape[0]):
        p = (tokenizer.decode(pred[i], skip_special_tokens=True)).split(" ")
        t = (tokenizer.decode(target[i], skip_special_tokens=True)).split(" ")
        bleu += sentence_gleu([t], p)
    return p, t, bleu/pred.shape[0]


def get_corpus_gleu(pred, target, max_length=512):
    bleu = 0
    for i in range(pred.shape[0]):
        p = clean_repeats(tokenizer.decode(pred[i], skip_special_tokens=True)).split(" ")
        t = clean_repeats(tokenizer.decode(target[i], skip_special_tokens=True)).split(" ")
        bleu += corpus_gleu(t, p)
    return p, t, bleu/pred.shape[0]


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts[predicts != -100], dim=-1)
    targets_prob = torch.nn.functional.softmax(targets[predicts != -100], dim=-1)
    return (- targets_prob * student_likelihood).mean()


def kl_div(predicts: torch.Tensor, targets: torch.Tensor, temperature=2.0):
    if len(predicts.shape) > 2:
        predicts = predicts.view(predicts.shape[0]*predicts.shape[1], predicts.shape[2])
        targets = targets.view(targets.shape[0]*targets.shape[1], targets.shape[2])
    loss = F.kl_div(
        input=F.log_softmax(predicts / temperature, dim=-1),
        target=F.softmax(targets / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


def cos_loss(x1, x2, dim=0):
    x1 = x1.view(x1.shape[0] * x1.shape[1], x1.shape[2])
    x2 = x2.view(x2.shape[0] * x2.shape[1], x2.shape[2])
    return (1 - torch.nn.functional.cosine_similarity(x1, x2)).mean(dim=dim)