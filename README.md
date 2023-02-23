# 概要
https://www.kaggle.com/competitions/bootcamp2023/overview

このコードは上記のコンテスト用に作成されたものです。

くずし字MNISTデータセットのStratified K-fold Cross Validationを実行可能です。

コマンドラインから、Epoch数、batch_size、fold数、モデルの選択が可能です。

デフォルトだと、ResNet18とConcNeXt_tiny、自身で作成したモデルの3種から選択可能です。

modelファイル内のMynetを編集することで自分で構築することも可能です。

また、入力に関しては、main.pyと同ディレクトリにinputフォルダを作成し、入手したデータセットをそこに配置してください。
