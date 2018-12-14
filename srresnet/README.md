# SRResNet + DRAGAN
## Summary

![here](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/makegirlsmoe.png)
- GeneratorにSRResNetを導入、Residual BlocksとPixel Shufflerを考慮
- DiscriminatorにもResidual Blocksを導入。またconditionalGANを仮定してACGANを導入。

## Usage
使用方法としては予め`image_path`に128✕128の画像を格納しておいて以下のコマンドを実行
```bash
$ python train.py
```
これで`./output`配下に画像があるepoch毎に生成される。`train_multi.py`はConditionalに画像を生成。`face_tag.py`にタグ情報を入れる必要がある。

## Result
私の環境で生成した画像を以下に示す。
![Result](https://github.com/SerialLain3170/Illustration-Generator/blob/master/srresnet/summary_a.png)
![Result](https://github.com/SerialLain3170/Illustration-Generator/blob/master/srresnet/conditional.png)
- 77 epoch目
- バッチサイズは64
- Adamの各パラメータとしてはα=0.0002、β1=0.5
- gradient penaltyの重みは0.5、Adversarial lossの重みは34.0
- 上記で示したネットワーク構造の図とは違い、Discriminatorのフィルターの数は全ての層において半分としている。
