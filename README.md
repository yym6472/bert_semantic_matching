# bert_semantic_matching
用预训练BERT实现语义匹配。

## 依赖环境
- python==3.6.5
- allennlp==0.9.0
- torch==1.3.1

## 运行

### 训练
```
python3 train.py --config_path ./config/bert.lcqmc.json --output_dir ./output/bert-lcqmc/
```

### 预测
```
python3 test.py --output_dir ./output/bert-lcqmc/
```

## 数据格式说明

- CSV格式，包含三列：sentence1、sentence2、label
- 要求能通过pandas.read_csv进行读取加载