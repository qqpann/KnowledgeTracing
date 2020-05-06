import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm


def irt_prob(difficulty: float, ability: float) -> float:
    c = 0.25
    return c + (1 - c) / (1 + math.exp(difficulty - ability))


class Question:
    def __init__(self, lo_name: str, lo_id, rank: float = None):
        self.lo_name = lo_name
        self.lo_id = lo_id
        if rank is None:
            rank = np.random.normal(loc=0.0, scale=1.0)
        self.rank = rank


class Student:
    def __init__(self, number: int):
        self.number = number
        # scale_max = 0.3
        self.intelligence = 0  # np.random.normal(loc=0.0, scale=1.0)

    def get_irt_prob(self, question: Question) -> float:
        # prob = irt_prob(question.rank, self.intelligence[question.lo_id])
        prob = irt_prob(question.rank, self.intelligence)
        return prob

    def answer(self, irt_prob: float):
        ans_rand = random.random()  # uniform distribution
        if ans_rand < irt_prob:
            return True
        else:
            return False

    def levelup(self):
        return
        self.intelligence = 1.1 * self.intelligence
        if self.intelligence > 1:
            self.intelligence = 1


def main(outpath: Path):
    STUDENT_NUMS = 400

    students = []
    for num in range(STUDENT_NUMS):
        s = Student(num)
        students.append(s)

    questions = []
    for lo in range(5):
        for q in range(10):
            q = Question("lo{}_q{}".format(lo, q), lo,)
            questions.append(q)

    d: Dict[str, List] = {
        "SEQ": [],
        "CustomerNumber": [],
        "SkillID": [],
        "IRTprob": [],
        "AnswerResult": [],
    }

    seq = 0
    for s in tqdm.tqdm(students):
        for i, q in enumerate(questions):
            irt_prob = s.get_irt_prob(q)
            ans = s.answer(irt_prob)

            d["SEQ"].append(seq)
            d["CustomerNumber"].append(s.number)
            d["SkillID"].append(q.lo_name)
            d["IRTprob"].append(irt_prob)
            d["AnswerResult"].append(int(ans))
            # print("q: %s q_rank: %d  s_intelligence:%d ans:%d" % (q.lo_name, q.rank, s.intelligence, ans))
            # if i % 10 == 0:
            #     s.levelup()
            seq += 1

    history_df = pd.DataFrame(data=d)
    history_df = history_df.astype(
        {
            "SEQ": "int32",
            "CustomerNumber": "int32",
            "IRTprob": "float64",
            "AnswerResult": "int32",
        }
    )

    history_df.to_csv(outpath)


if __name__ == "__main__":
    outpath = Path(sys.argv[1])
    main(outpath)
