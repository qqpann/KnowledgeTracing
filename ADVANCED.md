# Knowledge Tracings

## Quick start

### Using docker

Docker version: 19.03

```terminal
git clone https://github.com/qqhann/KnowledgeTracing.git
cd KnowledgeTracing
make build
HOST_DIR={absolute path to KnowledgeTracing} make run
make exec
# You are now inside docker
python main.py config/{your-experiment}.json
```

## Parameter searching

### Optuna

Create config file ended with `*.optuna.json`.
Specify the range you want to explore by giving the range as list.

**Example**:

To explore `sequence_size` between 10 and 100, create a config file like below.

```json
// config/debug/optimize.optuna.json
{
  "debug": true,
  "cuda": true,
  "model_name": "ksdkt",
  "load_model": "",
  "overwrite": true,
  "source_data": "assist2009",
  "n_skills": 110,
  "sequence_size": [10, 100],
  "waviness_l1": true,
  "waviness_l2": true,
  "lambda_l1": 0.0,
  "lambda_l2": 0.0,
  "ksvector_l1": 0.5
}
```

And run it with the following command:

```bash
python optimize.py config/debug/optimize.optuna.json
```

## SageMaker

You may use AWS Sagemaker to run your experiments.

### [Input](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)

- `/opt/ml/input/data/{channel_name}` channel_name is set by user when creating SM Algorithm. S3 data will be copied here.

### [Output](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html)

- `/opt/ml/output/failure` Failure output should be here
- `/opt/ml/model` (Directory) Model checkpoints should be here.
