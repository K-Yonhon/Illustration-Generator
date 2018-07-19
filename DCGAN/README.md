# DCGAN

## Summary
![DCGAN](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/dcgan.png)

GANと同じ損失関数であるが、ネットワーク構造を変えている。具体的な変更点は以下の通り
- DiscriminatorとGeneratorどちらもPooling層は全てConvolution層に置き換える。
- DiscriminatorとGeneratorどちらもBatchNormalizationを追加。
- 隠れ層に全結合層は用いない。
- 活性化関数としてはGeneratorにはReLU(出力層ではTanh)、DiscriminatorにはLeaky-ReLUを用いる。
- 最適化手法としてはAdam(α = 0.0002, β1 = 0.5)を用いる。

以上のような変更点を加えることで64 * 64サイズの画像生成が可能になった。しかし以下のような問題点が生じる(これは論文中には明記されてないが後に分かったことである)。
- 更に解像度大きくしようとすると、似た画像が違うノイズから出力されるmode collapseが生じる。
- 学習が安定しない。

## Usage
使用方法としては、予め`.npy`ファイルを用意しておいて以下のコマンドを実行。
```
$ python dcgan.py
```
`image_out_dir`で指定したディレクトリにあるepoch毎(`interval`で指定)に画像を生成する。  

## Result
私の環境で生成した画像を以下に示す。
![image](https://github.com/SerialLain3170/Illustration-Generator/blob/master/DCGAN/result.png)
- 600 epoch目
- バッチサイズは100
- Adamの各パラメータとしては、α=0.0001,β=0.5
