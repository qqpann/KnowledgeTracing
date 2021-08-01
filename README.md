# Knowledge Tracing

## Quick start

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

## Datasets

We used prepared data by Zhang et al. <https://github.com/jennyzhang0215/DKVMN>

<!-- Table created here: https://www.tablesgenerator.com/markdown_tables# -->

| Dataset name                            | KC size | Link                                                                                   |
| --------------------------------------- | ------- | -------------------------------------------------------------------------------------- |
| ASSISTments Skill builder 2009-2010     | 110     | https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data           |
| ASSISTments Skill builder 2015          | 100     | https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data |
| ASSISTments Datamining competition 2017 |         | https://sites.google.com/view/assistmentsdatamining/home?authuser=0                    |
| statics                                 | 1223    |                                                                                        |
| synthetic                               | 50      |                                                                                        |

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
