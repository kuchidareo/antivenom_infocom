# PyTorch CPU MaxPool2dの処理順序

実験対象をPyTorch 2.13.0、CPU、float32、dense contiguous NCHW、forward、
`kernel=2`、`stride=2`、`padding=0`、`dilation=1`、1 threadに固定する。

一次資料：

- https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/Pooling.cpp
- https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/DilatedMaxPool2d.cpp
- https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/MaxPoolKernel.cpp

## Dispatch

通常dense tensorの`max_pool2d`は`max_pool2d_with_indices`を経由し、CPU dispatch
stubから`MaxPoolKernel.cpp`へ到達する。quantized、明示的MKLDNN tensor、mobile
XNNPACKは別経路なので本実験の対象外である。indexをPythonへ返さない場合も、通常
dense経路は内部でvalueとindexを計算する。

## NCHWのloop順序

contiguous kernelはNとCを`plane=n*C+c`へflattenし、planeをparallelizeする。
各chunk内のsource loop順序は、

```text
plane(n,c) -> output row oh -> output column ow
           -> kernel row ih -> kernel column iw -> comparison
```

である。window内では横方向`iw`が最内側なので、2x2は、

```text
x1 x2
x3 x4
```

のraster orderとなる。複数threadでは各coreが異なるplane chunkとpredictor履歴を
持つため、単一系列として扱う最初の実験は1 threadに限定する。

## 各windowは4比較

PyTorchはfloatのrunning maxを`-infinity`で初期化し、最初の値を含む全要素を
`value > maxval || isnan(value)`で比較する。

```text
B1 = 1[x1 > -inf or isnan(x1)]
B2 = 1[x2 > current_max or isnan(x2)]
B3 = 1[x3 > current_max or isnan(x3)]
B4 = 1[x4 > current_max or isnan(x4)]
```

有限値の実験ではB1は必ず1なので、実行系列は、

```text
1,b2,b3,b4, 1,b2,b3,b4, ...
```

となる。以前の`b2,b3,b4`だけのモデルはPyTorch順序ではないため廃止した。

同値は更新しないので先に出現したmaxが残る。NaNは無条件で更新する。現在の
controlled experimentは非有限値を拒否するが、ReLUによる同値ゼロは保持する。

## Padding・stride・dilation

開始位置は`ih0=oh*strideH-padH`、`iw0=ow*strideW-padW`で、範囲外部分を明示的な
padding値として比較せず、有効input座標だけを走査する。paddingありではborder
windowの比較数が変わり得るため、4比較固定モデルはpadding 0だけに使う。

strideがkernelより小さい場合は隣接windowがinputを共有し、window間依存性が
変わる。dilationを変えると同じkh/kw順でも参照座標が変わる。

## Channels-lastは別物

channels-last kernelは`n -> oh -> ow -> kh -> kw -> channel vectors`で処理する。
チャネル本体はSIMD比較maskと`blendv`で更新され、通常のscalar conditional branch
ではない。端数チャネルの三項演算も機械語branchになる保証はない。NCHW scalarの
Markov/PMU結果をchannels-lastへ一般化しない。

## Sourceと機械語

C++の`if`はconditional jump、conditional move、max命令などへ変換され得る。
本実験は上記source順序を再現する小kernelをGCCで作り、PMU測定前にdisassemblyで
浮動比較から条件jumpへflagsが渡ることを確認する。PyTorch wheelそのものへ拡張する
場合は、versionを固定した`libtorch_cpu.so`の対象kernel PCを別途同定する。
