# Spectral Normalization for GAN

## Summary
- DiscriminatorとGeneratorの各層にSpectral Normalizationを加える。これにより1-Lipchitzの条件を満たす。
- 論文中でも述べられている通り、ハイパーパラメータのチューニングを実質的にしなくてよい(チューニングする余地があるのはk-Lipchitzのkのみ)
- 従来のGANとの組み合わせが可能。

## Usage
予め`.npy`ファイルを作成した状態で以下のコマンドを実行。
```bash
$ python sn.py
```

## Result
私の環境で生成した画像を以下に示す。
![image](https://github.com/SerialLain3170/Illustration-Generator/blob/master/spectral_normalization/result.png)
- 530 epoch目
- バッチサイズ100
- 最適化手法はAdam(α=0.0001,β1=0.5)
