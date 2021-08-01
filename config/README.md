# Config

実験の Config について．

## Fallback

`fallback.json`
このファイルがデフォルトの設定です．

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

- `cuda`: GPU を利用する
- `source_data`: 利用するデータのファイル名．前処理済みのデータであることが必要．
- `n_skills`: データに含まれるスキル数（knowledge concept; learning object）
- `epoch_size`: 最大エポック数
- `early_stopping`: このエポック数続けて向上がなければ実験を切り上げる
- `pre_dummy_epoch_size`: pre-train のエポック数
- `lr`: 学習率．
- `sequence_size`: 扱うシーケンス長の上限
- `pad`: padding を行ってシーケンス長に満たないものを学習に含める
