# heptabot-small

| architecture            | correction | JFLEG | CoNLL-2014 | bea   | time
| ----------------------- | ---------- | ----- | ---------- | ----- | ----
| T5 base v1.0            | 0.791      | 0.597 | -          | -     | 
| T5 base V1.1(teacher)   | 0.821      | 0.815 | -          | 65.66 | 7.3375s
| DistilT5 small(student) | 0.801      | 0.772 | -          | 49.56 | 2.6375s
| Quantized T5            | 0.812      | 0.792 | -          | 54.21 | 3.33