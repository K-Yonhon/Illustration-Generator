# DRAGAN

## Summary
![DRAGAN](https://github.com/SerialLain3170/GAN-papers/blob/master/makegirlsmoe/dragan.png)  
WGAN-GPで提案されたGradient Penalty項追加をベースに、学習データの周りの領域においてのみ勾配に制約をかける。

## Usage
予め`.npy`ファイルを用意しておき、以下のコマンドを実行。
```bash
$ python dragan.py --type <GAN_TYPE>
```
GAN_TYPEには`Normal`か`RaGAN`を指定する。
- `Normal` : DCGANベースのDRAGAN
- `RaGAN` : DCGANベースだが、Relativistic average discriminatorを考慮  

私の環境では、２つのGAN_TYPEに違いは見受けられなかったので`Normal`で良いと思われる。

## Result
私の環境で生成した画像を以下に示す。(GAN_TYPEには`Normal`を指定)  
![image](https://github.com/SerialLain3170/Illustration-Generator/blob/master/DRAGAN/result.png)
- 600 epoch目
- バッチサイズは100
- 最適化手法はAdam(α=0.0001, β1=0.5)
- Gradient Penalty項の重みは10.0
