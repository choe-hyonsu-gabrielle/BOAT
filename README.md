# BOAT: to Bert Or not to bert thAt is The question
Transformer-based BERT-like pretrained character level Masked Language Model on private purpose.
(It's very similar to *ALBERT* but much smaller and simpler.)

### Features
- Pretrained character-level tokenizer with explicit white-spacing tokens & long tail truncation
- Cross-layer parameter sharing (cf. *ALBERT*)
- Factorized embedding parameterization (cf. *ALBERT*)
- Dynamic masking (totally random. cf. *RoBERTa*)
- Dynamic stripping on the inputs longer than `max_length`
- Sentence Order Prediction (SOP) task (cf. *ALBERT*; to be applied.)