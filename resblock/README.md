# Residual Blocks
## Summary

![here](https://github.com/SerialLain3170/Illustration-Generator/blob/master/resblock/model.png)

- 特に論文がある訳ではないが、GANの高解像度生成にはResidual Blocksを主体とした構造がよく用いられている
- 上記の図は、[projection discriminator](https://arxiv.org/pdf/1802.05637.pdf)から
- upsamplingの仕方は、Nearest Neighbor Upsampling -> Conv
- 当初は損失関数にgradient penaltyを導入していたが、安定しなかったため外した

## Usage
使用方法としては予め`image_path`に128✕128の画像を格納しておいて以下のコマンドを実行
```bash
$ python train.py
```
これで`./output`配下に画像があるepoch毎に生成される。

## Result
私の環境で生成した画像を以下に示す。
![Result](https://github.com/SerialLain3170/Illustration-Generator/blob/master/resblock/summarize.png)
![Result](https://github.com/SerialLain3170/Illustration-Generator/blob/master/resblock/conditional.png)
- 70 epoch目
- バッチサイズは16
- 少ないバッチサイズのため、上図のように不安定な絵になっている。
- Adamの各パラメータとしてはα=0.0002、β1=0.5、β2=0.99
