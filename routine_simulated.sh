python src/simulate.py data/raw_input/simulated5_reproduction_v$1.csv
python data/prepare_input_data.py --file simulated5_reproduction_v$1 -csid SkillID -cusr CustomerNumber -cans AnswerResult --rerun
echo '
{
  "model_name": "ksdkt",
  "source_data": "simulated5_reproduction_v'$1'",
  "n_skills": 50,
  "pad": true
}
' > config/simulated5_reproduction/20_0429_reproduction_v$1.json

python main.py config/simulated5_reproduction/20_0429_reproduction_v$1.json