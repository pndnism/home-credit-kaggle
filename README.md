## データの配置

本リポジトリを使用する際は、`scripts`と同じ階層に`data`ディレクトリを作成し、その中に`train`および`test`データを配置してください。

ディレクトリ構造は以下のようになります：
.  
├── data  
│   ├── train  
│   └── test  
└── scripts  

## Weights & Biasesの設定

本リポジトリでは、Weights & Biasesを使用して実験の管理を行っています。Weights & Biasesの設定方法については、[公式ドキュメント](https://www.wandb.jp/)を参照してください。

## リポジトリの運用方法

本リポジトリでは、実験を回すごとにcommitを切ることを推奨します。これにより、各実験の結果を効率的に追跡し、管理することが可能となります。