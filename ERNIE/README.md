## ERNIE: **E**nhanced **R**epresentation through k**N**owledge **I**nt**E**gration


### Fine-tuning 任务

在完成 ERNIE 模型的预训练后，即可利用预训练参数在特定的 NLP 任务上做 Fine-tuning。以下基于 ERNIE 的预训练模型，示例如何进行分类任务和序列标注任务的 Fine-tuning，如果要运行这些任务，请通过 [模型&数据](#模型-数据) 一节提供的链接预先下载好对应的预训练模型。

将下载的模型解压到 `${MODEL_PATH}` 路径下，`${MODEL_PATH}` 路径下包含模型参数目录 `params`;

将下载的任务数据解压到 `${TASK_DATA_PATH}` 路径下，`${TASK_DATA_PATH}` 路径包含 `LCQMC`、`XNLI`、`MSRA-NER`、`ChnSentCorp`、 `nlpcc-dbqa` 5个任务的训练数据和测试数据；

#### 单句和句对分类任务

1) **单句分类任务**:

 以 `ChnSentiCorp` 情感分类数据集作为单句分类任务示例，数据格式为包含2个字段的tsv文件，2个字段分别为: `text_a  label`, 示例数据如下:
 ```
label  text_a
0   当当网名不符实，订货多日不见送货，询问客服只会推托，只会要求用户再下订单。如此服务留不住顾客的。去别的网站买书服务更好。
0   XP的驱动不好找！我的17号提的货，现在就降价了100元，而且还送杀毒软件！
1   <荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦!
 ```

执行 `bash script/run_ChnSentiCorp.sh` 即可开始finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] ave loss: 0.189373, ave acc: 0.954167, data_num: 1200, elapsed time: 14.984404 s
[test evaluation] ave loss: 0.189387, ave acc: 0.950000, data_num: 1200, elapsed time: 14.737691 s
```

2) **句对分类任务**：

以 `LCQMC` 语义相似度任务作为句对分类任务示例，数据格式为包含3个字段的tsv文件，3个字段分别为: `text_a    text_b   label`，示例数据如下:
```
text_a  text_b  label
开初婚未育证明怎么弄？  初婚未育情况证明怎么开？    1
谁知道她是网络美女吗？  爱情这杯酒谁喝都会醉是什么歌    0
这腰带是什么牌子    护腰带什么牌子好    0
```
执行 `bash script/run_lcqmc.sh` 即可开始finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] ave loss: 0.290925, ave acc: 0.900704, data_num: 8802, elapsed time: 32.240948 s
[test evaluation] ave loss: 0.345714, ave acc: 0.878080, data_num: 12500, elapsed time: 39.738015 s
```

#### 序列标注任务

1) **实体识别**

 以 `MSRA-NER(SIGHAN 2006)` 作为示例，数据格式为包含2个字段的tsv文件，2个字段分别为: `text_a  label`, 示例数据如下:
 ```
 label  text_a
 在 这 里 恕 弟 不 恭 之 罪 ， 敢 在 尊 前 一 诤 ： 前 人 论 书 ， 每 曰 “ 字 字 有 来 历 ， 笔 笔 有 出 处 ” ， 细 读 公 字 ， 何 尝 跳 出 前 人 藩 篱 ， 自 隶 变 而 后 ， 直 至 明 季 ， 兄 有 何 新 出 ？    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
相 比 之 下 ， 青 岛 海 牛 队 和 广 州 松 日 队 的 雨 中 之 战 虽 然 也 是 0 ∶ 0 ， 但 乏 善 可 陈 。   O O O O O B-ORG I-ORG I-ORG I-ORG I-ORG O B-ORG I-ORG I-ORG I-ORG I-ORG O O O O O O O O O O O O O O O O O O O
理 由 多 多 ， 最 无 奈 的 却 是 ： 5 月 恰 逢 双 重 考 试 ， 她 攻 读 的 博 士 学 位 论 文 要 通 考 ； 她 任 教 的 两 所 学 校 ， 也 要 在 这 段 时 日 大 考 。    O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O
 ```

执行 `bash script/run_msra_ner.sh` 即可开始 finetune，执行结束后会输出如下所示的在验证集和测试集上的测试结果:

```
[dev evaluation] f1: 0.951949, precision: 0.944636, recall: 0.959376, elapsed time: 19.156693 s
[test evaluation] f1: 0.937390, precision: 0.925988, recall: 0.949077, elapsed time: 36.565929 s
```

### FAQ

#### 如何获取输入句子经过 ERNIE 编码后的 Embedding 表示?

可以通过 ernie_encoder.py 抽取出输入句子的 Embedding 表示和句子中每个 token 的 Embedding 表示，数据格式和 [Fine-tuning 任务](#Fine-tuning-任务) 一节中介绍的各种类型 Fine-tuning 任务的训练数据格式一致；以获取 LCQM dev 数据集中的句子 Embedding 和 token embedding 为例，示例脚本如下:

```
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=7

python -u ernir_encoder.py \
                   --use_cuda true \
                   --batch_size 32 \
                   --output_dir "./test" \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --data_set ${TASK_DATA_PATH}/lcqmc/dev.tsv \
                   --vocab_path config/vocab.txt \
                   --max_seq_len 128 \
                   --ernie_config_path config/ernie_config.json
```

上述脚本运行结束后，会在当前路径的 test 目录下分别生成 `cls_emb.npy` 文件存储句子 embeddings 和 `top_layer_emb.npy` 文件存储 token embeddings; 实际使用时，参照示例脚本修改数据路径、embeddings 文件存储路径等配置即可运行；

#### 如何获取输入句子中每个 token 经过 ERNIE 编码后的 Embedding 表示？

[解决方案同上](#如何获取输入句子经过-ERNIE-编码后的-Embedding-表示?)
