# MaxPool Markov-to-PMU experiment

PyTorch 2.13.0のCPU contiguous NCHW MaxPool2d比較系列をMarkovモデル化し、Linux
PMU branch missesとの対応をpaired controlで検証する。正確な処理順序と適用範囲は
最初に[ORDERING.md](ORDERING.md)を読むこと。argmax変更をbranch missとは呼ばない。

## Linuxでの手順

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x build.sh
./build.sh
python verify_disassembly.py ./maxpool_benchmark
python -m unittest -v test_markov.py
python generate_inputs.py --output-dir data --windows 200000
python run_linux_experiment.py --manifest data/manifest.json \
  --benchmark ./maxpool_benchmark --output results/raw.jsonl \
  --trials 10 --repeats 100 --cpu 2
python analyze_results.py --manifest data/manifest.json \
  --measurements results/raw.jsonl --output-dir results/analysis
```

PMUはC++内の`perf_event_open`でwarmup後のreplay区間だけ測る。branched/controlの
順序とcondition順をtrialごとにrandomizeする。主要responseは、

```text
(branched misses - control misses) / (4 * windows * repeats)
```

である。validity diagnosticとして、

```text
(branched retired branches - control retired branches) / target comparisons
```

が1に近いことを必須とする。離れていれば対象data branchを分離できていない。

出力はpaired trial CSV、condition summary CSV、model fit JSON。各Markov proxyに対し
Pearson/Spearman、uncalibrated error、線形calibration、leave-one-condition-out性能を
報告する。実CPU predictor構造を2-bitと断定せず、machine固有calibrationとして扱う。

実CNN featureは`[N,C,H,W]`の`.npy`を保存して、

```bash
python import_feature_map.py --input features.npy --output-dir data_real \
  --condition-id modelA_layer1_epoch10
```

で同じwindow順へ変換できる。単一traceでは関係を検証できないので、layer、epoch、
seed、datasetなど複数conditionを用意する。

この実験が支持できるのは、指定CPU・compiler・source-faithful scalar kernelにおいて
feature由来Markov統計がPMU miss率の変動を予測する、という限定的な主張である。
