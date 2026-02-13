# センチメント情報と深層距離学習を考慮した会話におけるマルチモーダル感情認識モデル
![Figure1](./Suggest.png)
モデルの全体構造

以下の操作を、順番に行ってください。
## リポジトリのクローン
```bash
git clone https://github.com/midoor801/SCL.git
```
## 実験環境
Ubuntu環境で、NVIDIA Driverおよびanacondaのインストールが行われていることを想定します。SCLディレクトリに移動して以下を実行してください。
1. python 3.9
```
conda create -n SCL python=3.9
```
2. requirements.txt
```
pip install -r requirements.txt
```
3. ffmpegのインストール
ffmpegがインストールされているか確認してください。
```
ffmpeg -version
```
インストールされていない場合は、以下のコマンドを実行してください。
```
sudo apt update
sudo apt install ffmpeg
```
## データセット
各データセットを[datasetディレクトリ](./dataset)に配置してください。データセットは下記のリンクからダウンロードしてください。
1. [MELD](https://affective-meld.github.io)
下記のコマンドを実行してください。
```
wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```
インストールできない場合は、以下のコマンドを実行してください
```
wget --no-check-certificate 
https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
```
3. [IEMOCAP](https://sail.usc.edu/iemocap/)
上記のリンクのreleaseページに従い、データのリクエストを行ってください。入力したメールアドレスに返信が来るので、データを習得してください。ここでは、データのうち、IEMOCAP_full_release.tarファイルを使用します。
