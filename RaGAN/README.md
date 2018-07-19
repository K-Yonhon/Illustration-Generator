# Relativistic GAN

## Summary
![RGAN](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/rgan.png)  

- 真のデータと偽のデータのスコア差を取って学習するRelativistic discriminatorを提案。  
- 真のデータ、偽のデータのどちらかの平均を取って、もう片方との差を学習するRelativistic average discriminatorも同様に提案している。  
  - 論文中では後者の方がFIDは低くなっている。
- 単にdiscriminatorに代入する引数が変わっただけなので従来のGANとの組み合わせが可能。
- 論文中ではDCGANベースで256 * 256の鮮明な画像の生成に成功している。

## Usage
予め`.npy`ファイルを作成した状態で以下のコマンドを実行。
```bash
$ python train.py --type <GAN_TYPE>
```
GAN_TYPEには以下の３つが使用可能。
- `DCGAN` : 単なるDCGAN。Relativistic discriminatorは考慮していない。
- `RGAN` : DCGANをベースにRelativistic discriminatorを考慮。
- `RaGAN` : DCGANをベースにRelativistic average discriminatorを考慮。

また
```bash
$ python train.py --type <GAN_TYPE> --CR True
```
とすることでCritical Regularizer(ゼロ中心勾配の項に制約をかけたもの)を追加できる。

## Result
私の環境で生成した画像を以下に示す。
![image](https://github.com/SerialLain3170/Illustration-Generator/blob/master/RaGAN/result.png)
- 画像サイズは128 * 128
- GAN_TYPEは`RaGAN`
- Critical Regularizerは考慮
- 170 epoch目
- バッチサイズは100
- 最適化手法はAdam(α=0.0001, β1=0.5)
- 制約項の重みは10.0
