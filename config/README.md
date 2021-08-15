# Config

About experimental config JSON files.

## Fallback

`fallback.json`
This is the fallback config containing default fields that any experiment will fallback to if a field is not overwritten.

## Params

```json
{
  "debug": false,
  "cuda": true,
  "model_name": null,
  "source_data": "{{YOUR_DATASET_NAME}}",
  "n_skills": 100, // Your skill num
  "test_size": 0.2,
  "kfold": 5,
  "epoch_size": 2000,
  "early_stopping": 150,
  "pre_dummy_epoch_size": 0,
  "straighten_during_train_every": 0,
  "straighten_during_train_for": 0,
  "batch_size": 128,
  "lr": 0.05,
  "sequence_size": 20,
  "pad": false,
  "waviness_l1": true,
  "waviness_l2": true,
  "lambda_l1": 0.0,
  "lambda_l2": 0.0,
  "ksvector_l1": 0.0,
  "dkt": {
    "hidden_size": 200,
    "n_layers": 2,
    "bidirectional": false,
    "preserved_tokens": 2,
    "dropout_rate": 0.6
  }
}
```

- `cuda`: Use GPU if set true.
- `source_data`: The file name to use as source data. Requires preprocessed.
- `n_skills`: The number of skills (knowledge concept; learning object) included in the dataset.
- `epoch_size`: The maximum epoch size if not early stopped.
- `early_stopping`: How many epochs to keep training before early stopping.
- `pre_dummy_epoch_size`: How many epochs to do pre-training.
- `lr`: Leaning rate.
- `sequence_size`: Maximum sequence length to trim to.
- `pad`: If set true, use padding to handle sequences shorter than `sequence_size` after trimming.
