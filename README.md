# heptabot-small
T5 neural network pre-trained on correction, JFLEG, CoNLL-2014, and bea datasets. heptabot-small have state of the art results on JFLEG and correction datasets
.DistilT5 small is T5 base V1.1 distilled to the T5 small model.

| architecture            | correction | JFLEG | CoNLL-2014 | bea   | time    | memory 
| ----------------------- | ---------- | ----- | ---------- | ----- | ----    | ------
| T5 base v1.0            | 0.791      | 0.597 | -          | -     | -       | -
| T5 base V1.1(teacher)   | 0.821      | 0.815 | -          | 65.66 | 7.3375s | 990.4 MB
| DistilT5 small(student) | 0.809      | 0.772 | -          | 49.56 | 2.6375s | 307.9 MB
| Quantized T5            | 0.812      | 0.792 | -          | 54.21 | 3.3333s | 322.0 MB
| TinyT5                  | 0.814      | 0.780 | -          | 49.53 | 2.6833s | 307.9 MB
| Quantized TinyT5        | 0.785      | 0.698 | -          | 46.98 | 0.9193s | 126.5 MB
| SOTA                    | -          | 0.623 | 0.6          | 65.36 | -       | -

### training 
python train.py --path $PATH_TO_DATA --dir $DIR_FOR_SAVING --tokenizer $TOKENIZER --model $PATH_TO_MODEL --student $PATH_TO_STUDENT

