# Encoder-Decoder DKT

## Quick start

前提：
Docker環境またはEnviromnentsに書かれた環境が整備できていること．
GitHubアカウントの認証が通ること．

### Git clone
```
git clone git@github.com:qqhann/KnowledgeTracing.git
```
または
```
git clone https://github.com/qqhann/KnowledgeTracing.git
```

### Docker start
```
cd KnowledgeTracing
pwd
```
pwdで絶対パスが出力される．これを仮に`/home/zlt/KnowledgeTracing`だとする．（環境により異なる）

```
make build
HOST_DIR=/home/zlt/KnowledgeTracing make run
make exec
```
makeコマンドの命令内容はMakefileにより管理されている．
make execによってdocker内に入ることができた．

```
python main.py config/my-experiment.json
```
docker内では先ほどのmake buildで環境が整っているので，そのままpythonを実行できる．
configディレクトリ以下に`my-experiment.json`という設定ファイルがあると仮定して，その設定に基づいて学習を行う．

### Python start
Dockerを用いずにPythonなどの環境をEnvironmentsに書かれているように設定できている場合．
```
pip install -r requirements.txt
```
これによりPythonの依存ライブラリをインストール．

```
python main.py config/my-experiment.json
```
後はDocker内に入った場合と同様に実行する．

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
