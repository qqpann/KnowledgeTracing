# Knowledge Tracing

## Quick start

### Using poetry

```terminal
poetry install
poetry shell
# Alternatively, you can use pip instead of poetry
# pip install -r requirements.txt
python main.py config/debug/debug.json
```

Create your own config JSON file and you can start your experiment.

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
| PyTorch      | 1.5.0   |
| Docker       | 19.03   |

## BibTex

Please cite our paper if you use the code.

### Pre-training

To reproduce, read `notebook/Results_EDM2021.ipynb` and run the same experiment.

```bibtex
# To be added
```

Title: Prior Knowledge on the Dynamics of Skill Acquisition Improves Deep Knowledge Tracing
Author: Qiushi Pan & Taro Tezuka
Conference: ICCE2021

### Knowledge State Vector Loss

To reproduce, read `notebook/Results_EDM2020.ipynb` and run the same experiment.

```bibtex
@article{panaccuracy,
  title={Accuracy-aware Deep Knowledge Tracing with Knowledge State Vector Loss},
  author={PAN, Qiushi and TEZUKA, Taro}
}
```

## References

- DKT by author Piech <https://github.com/chrispiech/DeepKnowledgeTracing>
- DKVMN by author Zhang <https://github.com/jennyzhang0215/DKVMN>
- DeepIRT by author Yeung <https://github.com/ckyeungac/DeepIRT>
