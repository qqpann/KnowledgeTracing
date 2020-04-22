import os
import codecs
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold

import click


# Process into Sequence by user
def id_generator():
    n = 0
    while True:
        yield n
        n += 1


def ask_for(obj: str, message: str):
    ''' Ask for input if not provided '''
    if obj:
        return obj
    return input(message)


def dump_qa_fmt(obj: Dict, f) -> None:
    for usr, seq in obj.items():
        f.write('{}\n'.format(usr))
        problems = ','.join([str(qa[0]) for qa in seq])
        answers = ','.join([str(qa[1]) for qa in seq])
        f.write('{}\n'.format(problems))
        f.write('{}\n'.format(answers))


@click.command()
@click.option('--file', 'infile', default='skill_builder_data_corrected', help='File name without extension')
@click.option('-csid', '--col-sid', 'csid', default='', help='Column name of skill id')
@click.option('-cusr', '--col-usr', 'cusr', default='', help='Column name of user id')
@click.option('-cans', '--col-ans', 'cans', default='', help='Column name of answer')
@click.option('--sort-by', default='', help='Column name to sort by')
@click.option('--folds', default=5, help='Validation k-folds to split into')
@click.option('--test-size', default=.2, help='Test data size to split')
@click.option('--split-mode', default='user', help='How to split valid or test')
@click.option('--outfile-name', default='', help='File name to output')
@click.option('--rerun', is_flag=True, help='Overwrite the output')
def main(infile, csid, cusr, cans, sort_by, folds, test_size, split_mode,  outfile_name, rerun):
    run(infile, csid, cusr, cans, sort_by, folds, test_size, split_mode, outfile_name, rerun)


def run(infile, csid, cusr, cans, sort_by, folds, test_size, split_mode, outfile_name='', rerun=False):
    """
    TODO: support random split
    TODO: support time split
    """
    dirname = Path(os.path.dirname(__file__))
    if (dirname / infile).exists():
        # Bad solution. Accepts path --file argument.
        infile = (dirname / infile).stem
    infname = dirname / f'raw_input/{infile}.csv'
    assert infname.exists(), f'{infname} not found.'
    if not outfile_name:
        outfile_name = infile
    outdir = dirname / f'input/{outfile_name}'
    if outdir.exists() and not rerun:
        print('Aborting: Outdir exists and not rerun')
        print('No change will be made')
        return
    if not outdir.exists():
        outdir.mkdir(parents=False, exist_ok=False)

    try:
        with codecs.open(infname, 'r', 'utf-8', 'strict') as f:
            df = pd.read_csv(f)
            print(df.columns)
    except Exception as e:
        print('HINT: Encoding problem. Use commands such as `nkf` to change to utf8 beforehand.')
        print('example: nkf -Lu -w file_name > new_file_name')
        print(e)

    print(df.shape)
    csid = ask_for(csid, 'Column name of skill id? > ')
    cusr = ask_for(cusr, 'Column name of user id? > ')
    cans = ask_for(cans, 'Column name of answer? > ')
    df.dropna(subset=[csid, cusr, cans])
    print(df.shape)
    assert set(df[cans].unique()) == {0, 1}, f'cans consists of {df[cans].unique()}. Only [0, 1] are supported'

    if sort_by:
        df = df.sort_values(by=sort_by)

    p_it = iter(id_generator())
    u_it = iter(id_generator())
    users = defaultdict(lambda: next(u_it))
    problems = defaultdict(lambda: next(p_it))
    processed = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        sid = row[csid]
        usr = row[cusr]
        ans = int(row[cans])
        # processed[row.user_id].append((problems[row.problem_id], row.correct))
        processed[users[usr]].append((problems[sid], ans))

    print('Knowledge Concepts:', len(problems))
    print('Students:', len(users))
    assert len(processed) == len(users), f'{len(processed)} and {len(users)} mismatch'

    all_idx = list(users.values())
    train_idx, test_idx = train_test_split(all_idx, test_size=test_size)
    kf = KFold(n_splits=folds, shuffle=False, random_state=None)

    outdicname = outdir / f'{outfile_name}_dic.pickle'
    with open(outdicname, 'wb') as f:
        pickle.dump(dict(problems), f)

    with open(outdir / f'{outfile_name}_train.txt', 'w') as f:
        result = {k: v for k, v in processed.items() if k in train_idx}
        assert len(result) > 0
        dump_qa_fmt(result, f)
    with open(outdir / f'{outfile_name}_train.pkl', 'wb') as f:
        result = {k: v for k, v in processed.items() if k in train_idx}
        assert len(result) > 0
        pickle.dump(result, f)

    with open(outdir / f'{outfile_name}_test.txt', 'w') as f:
        result = {k: v for k, v in processed.items() if k in test_idx}
        assert len(result) > 0
        dump_qa_fmt(result, f)
    with open(outdir / f'{outfile_name}_test.pkl', 'wb') as f:
        result = {k: v for k, v in processed.items() if k in test_idx}
        assert len(result) > 0
        pickle.dump(result, f)

    for k, (train_k_idx, valid_k_idx) in enumerate(kf.split(train_idx), start=1):
        with open(outdir / f'{outfile_name}_train{k}.txt', 'w') as f:
            result = {k: v for k, v in processed.items() if k in train_k_idx}
            assert len(result) > 0
            dump_qa_fmt(result, f)
        with open(outdir / f'{outfile_name}_valid{k}.txt', 'w') as f:
            result = {k: v for k, v in processed.items() if k in valid_k_idx}
            assert len(result) > 0
            dump_qa_fmt(result, f)


if __name__ == '__main__':
    main()
