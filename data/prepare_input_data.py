import codecs
import os
from collections import defaultdict
import pickle

import pandas as pd
import click


# Process into Sequence by user
def id_generator():
    n = 0
    while True:
        yield n
        n += 1


@click.command()
@click.option('--file', default='skill_builder_data_corrected', help='File name without extension')
@click.option('-csid', '--col-sid', 'csid', default='skill_id', help='Column name of skill id')
@click.option('-cusr', '--col-usr', 'cusr', default='user_id', help='Column name of user id')
@click.option('-cans', '--col-ans', 'cans', default='correct', help='Column name of answer')
@click.option('--sort-by', default='', help='Column name to sort by')
@click.option('--outfile-name', default='', help='File name to output')
def main(file, csid, cusr, cans, sort_by, outfile_name):
    run(file, csid, cusr, cans, sort_by, outfile_name)


def run(file, csid, cusr, cans, sort_by, outfile_name=''):
    dirname = os.path.dirname(__file__)
    infname = os.path.join(dirname, f'raw_input/{file}.csv')
    outfname = os.path.join(dirname, f'input/{outfile_name if outfile_name else file}.pickle')
    outfname_txt = os.path.join(dirname, f'input/{outfile_name if outfile_name else file}.txt')
    outdicname = os.path.join(dirname, f'input/{outfile_name if outfile_name else file}_dic.pickle')
    # order_id,assignment_id,user_id,assistment_id,problem_id,original,correct,attempt_count,ms_first_response,tutor_mode,answer_type,sequence_id,student_class_id,position,type,base_sequence_id,skill_id,skill_name,teacher_id,school_id,hint_count,hint_total,overlap_time,template_id,answer_id,answer_text,first_action,bottom_hint,opportunity,opportunity_original
    print(infname)

    try:
        with codecs.open(infname, 'r', 'utf-8', 'strict') as f:
            df = pd.read_csv(f)
            print(df.columns)
    except Exception as e:
        print('HINT: Encoding problem. Use commands such as `nkf` to change to utf8 beforehand.')
        print('example: nkf -Lu -w file_name > new_file_name')
        print(e)


    print(df.shape)
    df.dropna(subset=[csid, cusr, cans])
    print(df.shape)

    df.dropna(subset=[csid, cusr, cans])

    if sort_by:
        df = df.sort_values(by=sort_by)

    it = iter(id_generator())

    processed = defaultdict(list)
    problems = defaultdict(lambda: next(it))
    for idx, row in df.iterrows():
        # nanは無視する
        sid = row[csid]
        usr = row[cusr]
        ans = row[cans]
        # processed[row.user_id].append((problems[row.problem_id], row.correct))
        processed[usr].append((problems[sid], ans))

    print('Problems:', len(problems))
    print('Students:', len(processed))

    # Save processed data
    with open(outfname, 'wb') as f:
        pickle.dump(dict(processed), f)
    with open(outdicname, 'wb') as f:
        pickle.dump(dict(problems), f)
    with open(outfname_txt, 'w') as f:
        for usr, seq in processed.items():
            f.write('{}\n'.format(usr))
            problems = ','.join([str(qa[0]) for qa in seq])
            answers = ','.join([str(qa[1]) for qa in seq])
            f.write('{}\n'.format(problems))
            f.write('{}\n'.format(answers))


if __name__ == '__main__':
    main()
