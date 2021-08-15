# Knowledge Tracing

## Quick start

### Install and run an experiment

```terminal
$ poetry install
$ poetry shell
$ python main.py config/debug/debug.json
```

Alternatively, you can use pip instead of poetry:
`pip install -r requirements.txt`.

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

## BibTex

Please cite our paper if you use the code.

### Pre-training

To reproduce, read `notebook/Results_ICCE2021.ipynb` and run the same experiment.

```bibtex
# To appear
@article{panaccuracy,
  title={Prior knowledge on the dynamics of skill acquisition improves deep knowledge tracing},
  author={Qiushi Pan and Taro Tezuka},
  booktitle={Proceedings of the 29th International Conference on Computers in Education, },
  year={2021}
}
```

```text
Qiushi Pan and Taro Tezuka, Prior knowledge on the dynamics of skill acquisition improves deep knowledge tracing, Proceedings of the 29th International Conference on Computers in Education, November 2021 (ICCE2021). (to appear)
```

### Knowledge State Vector Loss

To reproduce, read `notebook/Results_ICCE2020.ipynb` and run the same experiment.

```bibtex
@article{panaccuracy,
  title={Accuracy-aware Deep Knowledge Tracing with Knowledge State Vector Loss},
  author={Qiushi Pan and Taro Tezuka},
  booktitle={Proceedings of the 28th International Conference on Computers in Education (ICCE2020)},
  year={2020}
}
```

```text
Qiushi Pan and Taro Tezuka, Accuracy-aware deep knowledge tracing with knowledge state vectors and an encoder-decoder architecture, Proceedings of the 28th International Conference on Computers in Education (ICCE2020), Online, November 23-27, 2020.
```

<https://apsce.net/icce/icce2020/proceedings/paper_58.pdf>

## References

- DKT by author Piech <https://github.com/chrispiech/DeepKnowledgeTracing>
- DKVMN by author Zhang <https://github.com/jennyzhang0215/DKVMN>
- DeepIRT by author Yeung <https://github.com/ckyeungac/DeepIRT>
