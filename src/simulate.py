import math
import random
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm


class LevelUpMode(Enum):
  NORMAL = 1 # always levelup
  CORRECT = 2 # when correct asnwer only
  INCORRECT = 3 # when incorrect asnwer only
  BOTH = 4 # when incorrect asnwer: level down, when correct asnwer: level up 
  NONE = 5 # no level up

def irt_prob(difficulty: float, ability: float) -> float:
    c = 0.25
    return c + (1 - c) / (1 + math.exp(difficulty - ability))


class Question:
    def __init__(self, lo_name: str, lo_id: int, rank: float = None):
        self.lo_name = lo_name
        self.lo_id = lo_id
        if rank is None:
            rank = np.random.normal(loc=0.0, scale=1.0)
        self.rank = rank


class Student:
    def __init__(self, number: int):
        self.number = number
        self.init_intelligence()

    def init_intelligence(self):
        # scale_max = 0.3
        self.intelligence = np.random.normal(loc=0.0, scale=1.0)

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
        self.intelligence = 1.05 * self.intelligence

    def leveldown(self):
        self.intelligence = 0.95 * self.intelligence


def main(outpath: Path, mode:LevelUpMode = LevelUpMode.NORMAL):
    STUDENT_NUMS = 400

    students = []
    for num in range(STUDENT_NUMS):
        s = Student(num)
        students.append(s)

    questions: List[Question] = []
    for lo in range(5):
        for qid in range(10):
            q: Question = Question("lo{}_q{}".format(lo, qid), lo)
            questions.append(q)

    d: Dict[str, List] = {
        "SEQ": [],
        "CustomerNumber": [],
        "SkillID": [],
        "IRTprob": [],
        "AnswerResult": [],
    }

    seq = 0
    summary_inc = 0
    summary_dec = 0
    for s in tqdm.tqdm(students):
        for i, q in enumerate(questions):
            irt_prob = s.get_irt_prob(q)
            ans = s.answer(irt_prob)

            d["SEQ"].append(seq)
            d["CustomerNumber"].append(s.number)
            d["SkillID"].append(q.lo_name)
            d["IRTprob"].append(irt_prob)
            d["AnswerResult"].append(int(ans))
            # print(
            #     "q: %s q_rank: %d  s_intelligence:%d ans:%d"
            #     % (q.lo_name, q.rank, s.intelligence, ans)
            # )

            if mode == LevelUpMode.NORMAL:
                s.levelup()
                summary_inc += 1
            if (mode == LevelUpMode.CORRECT or mode == LevelUpMode.BOTH) and ans == True:
                s.levelup()
                summary_inc += 1
            if (mode == LevelUpMode.INCORRECT or mode == LevelUpMode.BOTH) and ans == False:
                s.leveldown()
                summary_dec += 1

            if i % 10 == 0:
                s.init_intelligence()
            seq += 1
    print('Total increase:', summary_inc)
    print('Total decrease:', summary_dec)

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
    mode = LevelUpMode(int(sys.argv[2]))
    print('Mode:', mode)
    main(outpath, mode)
