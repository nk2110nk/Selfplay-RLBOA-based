# Selfplay-RLBOA-based

`RLBOA-based` の三者間 PPO 実装を保ったまま、永続 opponent pool、PFSP、
frozen PPO snapshot、定期評価、完全 checkpoint/resume を追加した自己対戦版です。
実行時に sibling repository を import しません。

## 何を維持し、何を追加したか

手元のベースラインと同じく、交渉は三者間 SAOP、学習器は Stable-Baselines3
PPO `MlpPolicy`、観測は直近/一つ前の三者の offer utility と relative time の7次元、
行動は `Discrete(10)` の絶対 target utility bin です。bid は既存の
`HardHeadedFrequencyModel` を使う RLBOA bidding strategy が生成し、acceptance は
ACnext 型です。80 steps で、合意報酬は learner utility、不成立は学習時 -1、評価時
0です。

元の `RLBOA-based` に対して次を追加・修正しました。

- episode ごとに2相手を persistent pool から順序付き抽選
- scripted / historical PPO snapshot / optional current self-play の混成
- learner が観測可能な agreement、learner utility、length だけによる PFSP
- 10,000 cumulative steps ごとの均衡化pool評価とsnapshot判定
- SB3 optimizerを含むmodel、RNG、pool、scheduleを含むatomic checkpoint
- explicit role ID、observer完全reset、learner手番境界、case1〜case3
- 16列TSVとcheckpoint条件を記録した評価manifest

## 論文版RLBOAとの相違

論文 *RLBOA: A Modular Reinforcement Learning Framework for Autonomous Negotiating
Agents* は二者交渉・tabular Q-learning・utility/timeの離散状態・utility binの相対移動・
終端報酬です。このrepositoryは論文の完全再現ではありません。既存研究コードとの比較を
優先し、三者間SAOPの7次元観測とSB3 PPO、絶対utility-bin行動を維持しています。
Frequency opponent model と ACnext の考え方は引き継いでいます。

## 自己対戦

既定source mixtureは current self-play 0.0、scripted 0.5、snapshot 0.5です。
snapshotがまだない場合は利用可能sourceへ自動的に再正規化します。

各entryの難易度は次です。

```text
raw = .45*(1-agreement_rate) + .40*(1-agent_utility)
      + .15*clip(mean_length/80, 0, 1)
confidence = matches/(matches+10)
difficulty = confidence*raw + (1-confidence)*.5
```

PFSPは `softmax(pfsp_alpha*difficulty)` と一様分布を `uniform_mix` で混合します。
opponent utility、social welfare、Nash productは学習報酬・PFSP・snapshot採用判定に
使いません。snapshotはepisode中固定、eval mode、gradientなしで、各slotが独立した
observerとopponent modelを持ちます。current self-playもepisode開始時にdeep copyします。

## Utility case

- `case1`: learner=utility1, opponent1=utility2, opponent2=utility3
- `case2`: learner=utility2, opponent1=utility3, opponent2=utility1
- `case3`: learner=utility3, opponent1=utility1, opponent2=utility2

個別スクリプトの既定はcase1です。`CASE_NAME=case2`または
`CASE_NAME=case3`で同じ実験構成を実行できます。

## 学習

expert例:

```bash
python train.py -i Laptop -a Boulware Conceder --case case1 \
  --model-type expert --seed 0 --total-timesteps 100000
```

general例:

```bash
python train.py -i Laptop Car -a Boulware Linear Conceder \
  --model-type general --case case1 --seed 0 --total-timesteps 100000
```

`--total-timesteps` は追加量ではなく累積到達stepです。resumeは明示します。

```bash
python train.py --resume results/seed-0/case1/models/expert/Boulware-Conceder/Laptop/RLBOASelfPlay_Negotiator \
  --total-timesteps 200000
```

評価:

```bash
python test_negotiator.py --model <MODEL_DIR> --domain Laptop \
  --opponent1 Boulware --opponent2 Conceder --case case1 \
  --episodes 100 --seed 0
```

既定評価は stochastic、noiseなしです。`--deterministic` と `--noise` で条件を変更でき、
条件はファイル名とmanifestに保存されます。評価時にはdomain、case、action数、model種別、
checkpointとtraining-stateのstep整合性を検証します。
評価TSVとmanifestは各モデルディレクトリ内に保存されます。

```text
<model_dir>/csv/<opponent1>-<opponent2>/<domain>/det=False_noise=False/*.tsv
```

`--export-root` を指定した場合のみ、従来の集計用`evaluation/`にもコピーします。

## 一括スクリプト

expertの重複あり全組合せを処理します。個別スクリプトはcase1が既定で、
`CASE_NAME`でcase2/case3に切り替えられます。本学習は自動では開始されません。

```bash
DRY_RUN=1 ./run_command/train_case1_expert.sh
CASE_NAME=case2 DRY_RUN=1 ./run_command/train_case1_expert.sh
DOMAINS="Laptop" AGENTS="Boulware Conceder" SEED=1 ./run_command/train_case1_expert.sh
RESUME=1 ./run_command/train_case1_expert.sh
./run_command/test_case1_expert.sh

# case1〜case3を順番に実行
DRY_RUN=1 ./run_command/train_cases1_to3_expert.sh
./run_command/test_cases1_to3_expert.sh
```

`RESULTS_ROOT`, `SMOKE_ROOT`, `DOMAINS`, `AGENTS`, `SEED`, `TIMESTEPS`, `N_ENVS`,
`N_STEPS`, `BATCH_SIZE`, `CASE_NAME` を環境変数で変更できます。全case用スクリプトは
`CASES="case1 case3"`のように対象を絞れます。既存checkpointは既定でskipし、
`RESUME=1` のときだけ再開します。評価済みデータは行数ではなくmanifest中のcheckpoint
hash、seed、case、条件、episode数で検証します。

### case1〜3の本実験

`run_case_experiments.sh`は、seed 0でexpert 210モデル（3 cases × 7 domains ×
10 opponent pairs）とgeneral 3モデルを生成します。expertは100,000 steps、generalは
既知7ドメインをまとめて300,000 steps学習します。RLBOA固有のPPO設定は既定で
`n_envs=4`, `n_steps=500`, `batch_size=64`, `n_epochs=10`です。

```bash
# 生成されるコマンドの確認
DRY_RUN=1 ./run_command/run_case_experiments.sh

# 3 GPU用コンテナを起動済みの場合
GPU_IDS="0 1 2" \
CONTAINER_NAMES="selfplay-rlboa-training-gpu0 selfplay-rlboa-training-gpu1 selfplay-rlboa-training-gpu2" \
./run_command/launch_case_experiments_tmux.sh

tmux attach -t selfplay-rlboa-cases
```

3 GPUではjob indexを3で割った余りにより静的shardingし、各GPUが71モデルを担当します。
これは1モデルを複数GPUで学習するDDPではなく、独立モデルのジョブレベル並列です。
完了済みcheckpointはskipし、未完了checkpointは累積目標stepまでresumeします。

全学習完了後の評価はexpert 210条件とgeneral 360条件（3 cases × 12 domains ×
10 opponent pairs）の合計570条件、各100 episodesです。generalの12ドメインには既知7件と
未知5件（Coffee, Camera, Lunch, SmartPhone, Kitchen）を含みます。

```bash
DRY_RUN=1 ./run_command/evaluate_case_experiments.sh

MODE=eval GPU_IDS="0 1 2" \
CONTAINER_NAMES="selfplay-rlboa-training-gpu0 selfplay-rlboa-training-gpu1 selfplay-rlboa-training-gpu2" \
./run_command/launch_case_experiments_tmux.sh

tmux attach -t selfplay-rlboa-eval
```

実験条件は`RESULTS_ROOT`, `SEED`, `EXPERT_TIMESTEPS`, `GENERAL_TIMESTEPS`,
`N_ENVS`, `N_STEPS`, `BATCH_SIZE`, `N_EPOCHS`で変更できます。評価は`EPISODES`,
`FORCE`で変更できます。両スクリプトとも`SHARD_INDEX`と`SHARD_COUNT`を直接指定して
Dockerやtmuxを使わず実行することもできます。

## 保存構造

```text
results/seed-<seed>/<case>/models/<expert|general>/<pair>/<domain>/RLBOASelfPlay_Negotiator/
  checkpoint.zip
  training_state.pt
  config.json
  training_log.tsv
  tensorboard/
  pool/pool.json
  pool/stats.tsv
  pool/snapshots/
  evaluation/step-10000.tsv
  csv/

results/seed-<seed>/<case>/evaluation/<expert|general>/<pair>/<domain>/<case>/
```

別seedは別directoryへ保存され、上書きしません。`checkpoint.zip` はSB3 modelとoptimizer、
`training_state.pt` はcumulative step、config、pool metadata、RNG、次回評価/snapshot stepを
保持し、進行中の各NegMAS sessionとSB3の最終観測も保存します。pool中のsnapshot pathは
model directoryからの相対pathです。これによりrollout境界での中断・再開は、同じ依存環境
なら連続学習と同じ重みになります。依存versionも`config.json`へ記録し、異なるversionでの
resumeは拒否します。`--pool-path`はmodel directory内部だけを許可します。

## テスト

```bash
python -m pytest -q
```

## 既知の制約

- shared poolは単一processの`DummyVecEnv`を前提とします。SubprocVecEnvは使いません。
- resumeの再現性は同一依存version、device、`n_envs`、PPO設定を前提とします。
- evaluation/snapshot頻度は`n_envs * n_steps`の倍数である必要があります。
- evaluationは均衡化単位（domain × anchor × case × left/right slot）未満には減らしません。
- repository同梱requirementsは元ベースライン環境向けです。コードは検証用の
  NegMAS 0.9 / Gymnasium / SB3 2系にも互換層を持ちます。
