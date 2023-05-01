# Github Actionsについて

## 構造
```shell
.
├── .github
│   ├── workflows
│   │   ├── assign-pr-reviewers.yml     # レビュアー割り当てを決める
│   │   ├── check_directory_name.yml    # ディレクトリ名チェック
│   │   ├── checker.yml                 # コードチェック
│   │   ├── final_message.yml           # PR内の最後のメッセージ
│   │   ├── first_message.yml           # PR内の最初のメッセージ
│   │   └── set-pr-reviewers.yml        # 割り当てたレビュアーを実際に指名する
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── labeler.yml                     # 何回目の課題かを判別する用
├── ci
│   ├── assign_reviewers.py             # assign-pr-reviewersのworkflowで実行されるスクリプト
│   └── users.csv                       # 輪講参加者と割り当て情報を格納
├── docs
├── README.md
├── ex1
└── ...
```

## ラベル
どのディレクトリに対してpushしているかによって異なるラベルを付与される．

レビュアー割り当てで利用される．

## レビュアー割り当て
割り当てのステップは2段階ある．

1. 割り当てを決定しusers.csvに書き込む（手動）

Actions > Assign PR Reviewers > Run workflow から実行する．

![assign_workflow](./figs/assign_reviewers_workflow.png)

これによりPRが作成される．users.csvの新たな列に次の課題のレビュアー割り当てが追加される．問題なければマージする．

### users.csvの書き方
以下のように4列で書く．**最後は空行にすること．**
- 1列目：各自のgithubのアカウント名
- 2列目：学年（参考程度であり，プログラムでは使われない）
- 3列目：グループ（受講生はstudentとする．その他は任意の名前．それぞれのグループから各PRにレビュアーを1人ずつ選抜する．）
- 4列目：研究室名（参考程度であり，プログラムでは使われない）

```sh
github_account,position,group,laboratory
<account1>,b4,student,takeda-lab
<account2>,b4,student,toda-lab
<account3>,b4,student,toda-lab
...
<accountX>,m2,reviewer,takeda-lab
（空白）
```

2. PRに対してレビュアーを設定する（自動）

1でマージすることで，その後作成されたPRに対してusers.csvの情報に基づいてレビュアーが指名される．どの課題かはラベルから判断される．

## コーディングチェック
デフォルトではCIチェックを通らずともマージが可能である．これはGithubのリポジトリ設定から変更できる．

また，flake8などのlinter，formatterの項目も自由に設定できる．CIのymlファイルの内容を変更することで反映できる．

## ブランチの保護
masterまたはmainにマージする場合，間違えて操作しないように保護するのがオススメ．これもGithubのリポジトリ設定から変更できる．

例）　レビュアー1人からapproveされないとマージできないようにする
