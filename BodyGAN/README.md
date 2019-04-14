# BodyGAN(Silhouette-conditional Generative Adversarial Network)
## Summary
- 特に論文があるわけではないが、PSGANのようにシルエットを条件付けとして全身画像を生成
- 損失関数はAdversarial loss + Gradient Penalty, DiscriminatorにSpectral Normalizationを導入
- シルエットはFully-Convolutional Network(FCN)を用いて抽出している

## Result
私の環境で生成した結果を以下に示す。
![result](https://github.com/SerialLain3170/Illustration-Generator/blob/master/BodyGAN/result.png)
- バッチサイズは64
- 最適化手法はAdam(α=0.0002, β1=0.5)
- Adversarial loss, Gradient penaltyの重みはそれぞれ1.0と0.5
- Adversarial lossの計算にはHinge loss
