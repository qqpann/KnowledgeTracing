# Encoder-Decoder DKT

## Quick start

### Using docker

```terminal
git clone https://github.com/qqhann/KnowledgeTracing.git
cd KnowledgeTracing
make build
HOST_DIR={absolute path to KnowledgeTracing} make run
make exec
# You are now inside docker
python main.py config/{your-experiment}.json
```

### Using pip

```terminal
pip install -r requirements.txt
python main.py config/my-experiment.json
```

### Using poetry

```terminal
poetry install
poetry run python main.py config/my-experiment.json
```

## Advanced

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

### [Input](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)

- `/opt/ml/input/data/{channel_name}` channel_name is set by user when creating SM Algorithm. S3 data will be copied here.

### [Output](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html)

- `/opt/ml/output/failure` Failure output should be here
- `/opt/ml/model` (Directory) Model checkpoints should be here.

## Datasets

We used prepared data by Zhang et al. <https://github.com/jennyzhang0215/DKVMN>

|           | KC size |
| --------- | ------- |
| assist09  | 110     |
| assist15  | 100     |
| statics   | 1223    |
| synthetic | 50      |

<!-- Table created here: https://www.tablesgenerator.com/markdown_tables# -->

### Dataset links

#### ASSISTments Skill builder 2009-2010

https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data

#### ASSISTments Skill builder 2015

<https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data>

#### ASSISTments Datamining competition 2017

<https://sites.google.com/view/assistmentsdatamining/home?authuser=0>

## Environments

| requirements | version |
| ------------ | ------- |
| Python       | 3.7     |
| CUDA         | 10.1    |
| PyTorch      | 1.3.0   |
| Docker       | 19.03   |

## References

- DKT by author Piech <https://github.com/chrispiech/DeepKnowledgeTracing>
- DKVMN by author Zhang <https://github.com/jennyzhang0215/DKVMN>
- DeepIRT by author Yeung <https://github.com/ckyeungac/DeepIRT>
