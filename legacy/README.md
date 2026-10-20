# legacy / 05_Comments_NLP（原始项目快照）

本目录是 `Portfolio/projects/05_Comments_NLP` 的**完整代码快照**，用于保留项目演进溯源：

- `Tmall_Review_Crawling_and_LDA-Based_Topic_Clustering_Implementation.py`
- `thesis_project/SHU_comments_NLP.py`（SnowNLP + LDA + 词云）
- `douban_spider.py` / `rail_tickets_spider.py` / `online_community_userinfo_NLP.py`
- `thesis_project/postopic.html`、词云与思维导图

**它不再参与新系统的运行**。新系统在 `src/` 下重新实现了：

| 05 能力 | 新实现 |
| --- | --- |
| SnowNLP 整句情感 | `src/absa_engine.py` 四元组 ABSA |
| 单模型 | `src/tri_agent.py` VADER/FinBERT/LLM + SDI |
| 离线批处理 | `src/stream_processor.py` Kafka 实时流 |
| LDA 主题 | `src/graph_builder.py` Entity–Event 图谱 |
| Excel/CSV | `src/storage.py` Elasticsearch + 本地 JSONL |
| 静态图 | `src/api_server.py` + `reports/*.png` |

> ⚠️ 原脚本中的 Cookie、代理 IP、`font_path` 均为历史示例值，早已失效；请勿直接运行爬虫。复现新系统请回到仓库根目录，使用 `scripts/mock_producer.py` 生成数据。
