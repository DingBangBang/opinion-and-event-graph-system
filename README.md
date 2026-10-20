# 实时多 Agent 舆情分析与事件图谱系统

> **Real-time Multi-Agent Opinion Analysis & Event Graph System**
>
> 从 `Portfolio/projects/05_Comments_NLP` 的「离线 SnowNLP + LDA」升级为
> **Kafka 实时流 → ABSA 四元组 → Tri-Agent(SDI 路由) → Elasticsearch → Neo4j GraphRAG → FastAPI → 量化信号** 的完整闭环。

![tests](https://img.shields.io/badge/tests-22%20passed-brightgreen)
![python](https://img.shields.io/badge/python-3.10-blue)
![stack](https://img.shields.io/badge/Kafka%20%C2%B7%20ES%20%C2%B7%20Neo4j%20%C2%B7%20FastAPI-orange)
![status](https://img.shields.io/badge/demo-reproducible-success)

---

## 0. TL;DR

| 项目 | 结果 |
| --- | --- |
| 演示数据 | 544 条中英混合评论（72 小时，含一次「币安提现」负面爆发） |
| ABSA 输出 | **1,720 条四元组**（实体 / 方面 / 观点词 / 极性） |
| 流处理 | 544/544 成功，0 异常；滑动窗口捕获 **1 次负面突增告警** |
| Tri-Agent 路由 | LLM 路由率 **20.2%**，约 **79.8%** 评论由小模型直接处理 |
| 成本 | 按 DeepSeek 官方价估算，全量 LLM ≈ **$160 / 百万条**；本策略 ≈ **$32 / 百万条** |
| 图谱 | 11 个实体、544 个事件、1,497 条 `INVOLVES` 关系；282 个负面事件 |
| 量化 | 情感(t) 与收益(t+1) 相关 **r = 0.7662**，Granger **p < 0.0001**（模拟 K 线，方法验证） |
| 工程 | Docker Compose 编排 6 个服务；22 个 pytest 全绿；无外部依赖时可离线降级运行 |

> ⚠️ 量化部分默认使用**模拟 K 线**：在 t-1 情感上注入了已知的领先关系，用于验证估计器能否恢复真实结构，**不构成真实市场结论**。`--live` 可切换 Binance 公共 API。

---

## 1. 为什么重构：05_Comments_NLP 的局限

原项目（[`legacy/05_Comments_NLP`](./legacy/05_Comments_NLP)）已经完成了「采集 → 清洗 → 分词 → 情感 → 词云 → LDA」的文本挖掘闭环，但它是**单机批处理**，上线后面临四个硬约束：

1. **情感粒度太粗**：SnowNLP 只给整条评论一个 0~1 分数，无法回答「用户到底对提现、手续费还是客服不满」；
2. **模型单一、无仲裁**：词典/小模型在反讽、跨句、领域词上会错，但没有机制发现「自己不确定」；
3. **离线、无时间维度**：Excel/CSV 结果无法支撑分钟级预警，也做不了事件时间线；
4. **没有服务化**：分析结果不能被 API、看板、量化策略复用。

### 1.1 与 05 的逐项对照

| 维度 | 05_Comments_NLP | 本系统 |
| --- | --- | --- |
| 分析单元 | 整条评论 | **四元组** `(entity, aspect, opinion, polarity)` |
| 情感模型 | SnowNLP 单模型 | **VADER + FinBERT + LLM 委员会**，SDI 动态路由 |
| 成本控制 | 无 | **SDI > θ 才调用 LLM**，默认仅 20% 流量走 API |
| 处理模式 | 离线脚本 | **Kafka 实时流** + 滑动窗口告警 |
| 存储 | Excel / CSV | **Elasticsearch**（检索/聚合）+ **Neo4j**（关系/多跳） |
| 知识结构 | 词云 / LDA 主题 | **Entity–Event 异构图** + GraphRAG 多跳查询 |
| 输出 | 静态图表 | **FastAPI** + 量化信号 + 可审计路由日志 |
| 部署 | 本机 Python | **Docker Compose**（ZK/Kafka/ES/Neo4j/API/Processor） |
| 可观测性 | 无 | `/health`、`/routing/stats`、`/alerts`、JSONL 审计 |
| 质量保障 | 无测试 | **22 个 pytest**（ABSA/路由/图谱/量化/API） |
| 降级能力 | — | 无 ES/Neo4j/LLM/模型时自动降级，不阻断主流程 |

### 1.2 迁移映射

| 05 中的资产 | 在新系统中的去向 |
| --- | --- |
| `Tmall_Review_Crawling_...py` / `douban_spider.py` | `scripts/mock_producer.py` 取代爬虫，改为可复现的 Kafka 模拟源；爬虫保留在 `legacy/` 作为数据采集经验 |
| `thesis_project/SHU_comments_NLP.py`（SnowNLP + LDA） | 拆分为 `absa_engine.py`（细粒度）、`tri_agent.py`（委员会）、`graph_builder.py`（主题→事件图谱） |
| 词云 / `pyLDAvis` | `reports/absa_aspects.png`、`reports/graph_preview.png`：从「词频」升级为「方面 × 极性」「实体 × 事件」 |
| 论文中的情感思维导图 | `README` 的 Mermaid 架构图 + 六维度设计 |

---

## 2. 系统架构

```mermaid
flowchart LR
    subgraph INGEST["① 实时接入"]
        P["mock_producer.py<br/>评论生成器"]
        K[("Kafka<br/>raw_comments")]
        P -->|JSON| K
    end

    subgraph NLP["② 细粒度理解"]
        S["stream_processor.py"]
        A["ABSA Engine<br/>HF / DeepSeek / Lexicon"]
        V["VADER Agent<br/>词级"]
        F["FinBERT Agent<br/>句级"]
        L["LLM Agent<br/>跨句推理"]
        R{"SDI Router<br/>SDI > θ ?"}
        K --> S --> A
        A -->|四元组| R
        V --> R
        F --> R
        R -->|"否 (≈80%)"| D["采纳高置信小模型<br/>成本≈0"]
        R -->|"是 (≈20%)"| L
        L --> D
    end

    subgraph STORE["③ 存储与知识"]
        E[("Elasticsearch<br/>评论/决策/告警")]
        G[("Neo4j<br/>Entity–Event Graph")]
        W["滑动窗口<br/>负面突增检测"]
        S --> E
        S --> W --> E
        S --> GB["GraphBuilder"] --> G
        D --> AUD["路由审计<br/>JSONL + ES"]
    end

    subgraph SERVE["④ 服务与分析"]
        API["FastAPI<br/>趋势 / 事件 / 多跳 / 统计"]
        Q["quant_signal.py<br/>滞后相关 + Granger"]
        NB["notebook<br/>可视化"]
        E --> API
        G --> API
        E --> Q --> NB
        API --> U["看板 / 告警 / 策略"]
    end

    style R fill:#fff3cd,stroke:#d39e00
    style L fill:#f8d7da,stroke:#c82333
    style D fill:#d4edda,stroke:#28a745
    style G fill:#d1ecf1,stroke:#17a2b8
```

**成本优势的核心**：LLM 只处理「小模型吵架」的样本，且所有路由决策都写入审计表，可随时回放、调阈值、做 A/B。**GraphRAG 的价值**：把孤立的评论变成「实体—事件—方面」可追溯链路，支持「提现问题影响了哪些平台/时间段/其他实体」这类多跳问题。


---

## 3. 六维度实现详解

### 维度一：基于方面的细粒度情感分析（ABSA）

**目标**：把「币安提现三天没到账，客服还敷衍」拆成：

```json
[
  {"entity": "币安", "aspect_category": "withdrawal",
   "opinion_term": "没到账", "sentiment": -0.85, "label": "negative"},
  {"entity": "币安", "aspect_category": "customer_service",
   "opinion_term": "敷衍", "sentiment": -0.90, "label": "negative"},
  {"entity": "币安", "aspect_category": "settlement_speed",
   "opinion_term": "没到账", "sentiment": -0.85, "label": "negative"}
]
```

`src/absa_engine.py` 支持三种后端，接口完全一致：

| 后端 | 触发条件 | 模型/实现 | 适用场景 |
| --- | --- | --- | --- |
| `hf` | 已安装 `transformers` 且模型已缓存或允许下载 | `yangheng/deberta-v3-base-absa-v1.1`，输入 `(sentence, aspect)` | 中文/英文细粒度，离线推理 |
| `deepseek` | 配置 `DEEPSEEK_API_KEY` | DeepSeek JSON mode，一次返回全部四元组 | 隐式方面、跨句、复杂表达 |
| `lexicon` | 默认兜底 | 双语实体表 + 13 类方面词典 + 否定/程度副词规则 | CI、无网环境、冷启动 |

**方面体系**（`src/resources/lexicons.py`）：`withdrawal / deposit / fee / security / customer_service / app_experience / settlement_speed / kyc / market / liquidity / product / compliance / general`。

ABSA 的工程细节：

- 实体识别采用**最长别名优先 + 英文词边界**，避免 `eth` 命中 `ethereum`、`火币` 命中 `币`；
- 按中英文标点切分子句，子句内做**方面-观点词最近邻**匹配，否定词/程度副词在 5 字符窗口内生效（如「不太满意」→ 负向）；
- `event_id = sha1(comment_id + aspect)`，Kafka 重试不会产生重复事件；
- 输出同时保留连续分数 `sentiment ∈ [-1, 1]` 与离散标签，供阈值、加权、聚合使用。

运行：

```bash
# 规则后端（零依赖）
ABSA_BACKEND=lexicon python -m src.absa_engine

# HuggingFace（首次需允许下载）
pip install -r requirements-ml.txt
ABSA_BACKEND=hf ABSA_ALLOW_MODEL_DOWNLOAD=true python -c \
  "from src.absa_engine import ABSAEngine; print(ABSAEngine().analyze('币安提现失败').structured)"

# DeepSeek
ABSA_BACKEND=deepseek DEEPSEEK_API_KEY=sk-... python -m src.absa_engine
```

---

### 维度二：多 Agent 委员会与 SDI 动态路由

`src/tri_agent.py` 中三个 Agent 的职责与预算：

| Agent | 粒度 | 实现 | 典型延迟 | 单次成本 |
| --- | --- | --- | --- | --- |
| VADER Agent | 词级 | 英文用 `vaderSentiment`；中文用等价的否定/程度副词词典（VADER 本身不支持中文） | < 1 ms | ≈ 0 |
| FinBERT Agent | 句级 | `ProsusAI/finbert`；无模型时用**金融子词典**（刻意不认识「敷衍/失望」等非金融词，模拟领域模型盲区） | 1~20 ms | ≈ 0 |
| LLM Agent | 跨句/语篇 | DeepSeek / OpenAI 兼容 JSON mode，接收 ABSA 四元组 + 两模型分数 | 0.5~3 s | 按 token |

**SDI（Semantic Divergence Index）**：

$$
\mathrm{SDI} = 0.7 \times \frac{|\mathrm{VADER} - \mathrm{FinBERT}|}{2} + 0.3 \times \mathbb{1}[\text{label}_{V} \ne \text{label}_{F}]
$$

- 分数差项归一化到 `[0,1]`：两个模型都在 `[-1,1]`，最大差为 2；
- 标签冲突项给 0.3 的硬惩罚：方向相反本身比「差 0.1 分」更危险；
- 默认阈值 `SDI_THRESHOLD=0.45`：`SDI > 0.45` 才升级 LLM，否则选择**置信度更高的小模型**。

**路由记录**（每次必写，JSONL + Elasticsearch `tri_agent_decisions`）：

```json
{
  "comment_id": "c_9f2...",
  "sdi": 0.6036,
  "threshold": 0.45,
  "routed_to_llm": true,
  "selected_agent": "llm",
  "vader_score": -0.87,
  "finbert_score": 0.0,
  "llm_score": -0.55,
  "estimated_cost_usd": 0.00017,
  "avoided_cost_usd": 0.0,
  "rationale": "SDI=0.604 > 0.45，升级 LLM 仲裁；客服敷衍属于服务体验，整体负面"
}
```

**没有 API Key 时**：`LLM Agent` 会降级为「VADER + FinBERT + ABSA 加权共识」，`routed_to_llm` 仍为 `true`，`rationale` 明确标注 fallback，保证数据不丢、链路可演示。

**成本模型**（`config/settings.py` 可改价）：

```text
prompt_tokens     ≈ 220 + len(text) / 1.5
completion_tokens ≈ 90
cost = prompt/1e6 * $0.27 + completion/1e6 * $1.10   # DeepSeek 官方价示例
```

离线 demo 中 544 条评论、20.2% 路由率，跳过 434 次 LLM 调用，按上述价格影子成本约 **$0.071**；换算为百万条约 **$130** 的 avoided cost。实际生产按 20.2% 调用，约 **$32/百万条**，相比全量 **$160/百万条** 节省 **≈79.8%**。

---

### 维度三：实时流式数据处理管道

```mermaid
sequenceDiagram
    participant P as mock_producer
    participant K as Kafka(raw_comments)
    participant S as stream_processor
    participant A as ABSA
    participant T as Tri-Agent
    participant E as Elasticsearch
    participant G as Neo4j
    P->>K: 每 200ms 一条 JSON
    loop poll(max_records=50)
        K->>S: consumer.poll()
        S->>A: analyze(text, entity_hint)
        A-->>S: 四元组
        S->>T: route(text, absa)
        T-->>S: decision + SDI 审计
        S->>E: index opinion_comments
        S->>G: MERGE Event/Entity/INVOLVES
        S->>S: 滑动窗口负面占比
        S->>E: 条件满足 → index opinion_alerts
        S->>K: commit offset
    end
```

- **消费语义**：`enable_auto_commit=False`，每批处理完再 `commit()`，避免「先提交后失败」丢数据；毒消息单条隔离，不阻塞分区。
- **幂等**：ES 用 `comment_id` 作为 `_id`，Neo4j 用 `event_id` MERGE；重复消费不会膨胀。
- **滑动窗口告警**（`NegativeSpikeDetector`）：按实体维护 `deque[(timestamp, negative, sentiment)]`，窗口默认 300s、最少 8 条、负面占比 > 60%，冷却 120s；触发后打印并写入 `opinion_alerts`。
- **演示结果**：在 24 条「币安提现失败」集中爆发时触发：
  `「币安」近 5 分钟负面占比 100%（8/8），均值情感 -0.43，超过阈值 60%`。

运行：

```bash
# 本地：从 JSONL 回放（不依赖 Kafka）
python -m src.stream_processor --replay data/sample_comments.jsonl

# 本地：连 Kafka
python scripts/mock_producer.py --count 200 --rate 5
python -m src.stream_processor --from-beginning
```

---

### 维度四：事件与实体知识图谱（GraphRAG）

**Schema**（`src/graph_builder.py`）：

```mermaid
erDiagram
    ENTITY ||--o{ INVOLVES : "被涉及"
    EVENT  ||--o{ INVOLVES : "涉及"
    ENTITY {
        string name PK
        string type "exchange/asset/payment..."
        datetime updated_at
    }
    EVENT {
        string event_id PK
        string label
        datetime timestamp
        float sentiment
        string sentiment_label
        boolean negative
        string aspect_category
    }
    INVOLVES {
        string comment_id
        string aspect
        string aspect_label
        float sentiment
        string polarity
        string opinion
        datetime timestamp
    }
```

**写入逻辑**：先由 `event_detector.build_event` 生成事件（主方面取 |sentiment| 最大的四元组），再按每个四元组 `MERGE` 实体与 `INVOLVES` 边。边属性同时保留**时间戳**与**情感极性**，因此可以直接在边上做时间过滤和符号过滤。

```cypher
MERGE (ev:Event {event_id: $event_id})
SET ev.label = $label, ev.timestamp = datetime($timestamp),
    ev.sentiment = $sentiment, ev.negative = $negative
WITH ev
UNWIND $links AS link
MERGE (en:Entity {name: link.entity})
MERGE (ev)-[r:INVOLVES {aspect: link.aspect}]->(en)
SET r.timestamp = datetime($timestamp),
    r.sentiment = link.sentiment,
    r.polarity = link.polarity,
    r.opinion = link.opinion
```

**多跳查询**（输入「提现」而不是实体名也能查）：

```cypher
MATCH (seed:Entity)<-[sr:INVOLVES]-(se:Event)
WHERE seed.name = $query OR sr.aspect = $query
   OR sr.aspect_label = $query
   OR toLower(se.label) CONTAINS toLower($query)
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS s
MATCH path = (s)<-[:INVOLVES*1..2]-(ev:Event)
WHERE $negative_only = false OR ev.sentiment < -0.15
RETURN ev.label, ev.timestamp, ev.sentiment, path
ORDER BY ev.timestamp DESC LIMIT 50;
```

**业务价值**（不只是「存图」）：

1. **根因定位**：从「提现」出发，一次查询拿到所有平台/时间段的负面事件，直接对应运营工单；
2. **传染分析**：同一事件涉及多个实体（交易所 + USDT + 银行）时，2 跳可发现风险沿资金链扩散；
3. **时间线重建**：边上时间戳支持「先手续费抱怨 → 后提现失败 → 再客服投诉」的叙事；
4. **合规审计**：每条边可回溯到 `comment_id` 与原始文本，满足解释性要求；
5. **GraphRAG**：把子图作为 LLM 的检索上下文，回答「最近哪些事件导致币安负面情绪上升」这类多跳问题，而不是只做向量相似度检索。

实现上同时提供 `Neo4jGraphStore` 与 `InMemoryGraphStore`：Docker 环境用 Neo4j，单测/离线 demo 用内存图，两者返回相同的 `MultiHopResult`。

---

### 维度五：情感-价格相关性分析

`src/quant_signal.py` + [`notebooks/sentiment_price_correlation.ipynb`](./notebooks/sentiment_price_correlation.ipynb)。

1. 从 `processed.jsonl` 读取评论，按 `entity` 过滤、按小时重采样，得到 `sentiment_t`、`negative_ratio_t`、`volume_t`；
2. 获取价格：`live=True` 走 Binance `/api/v3/klines`（公开接口，无需 Key）；离线时用确定性 GBM 模拟；
3. 计算 `corr(sentiment_t, log_return_{t+k})`，`k = 0..12`，用 t 检验给 p 值；
4. 格兰杰检验（NumPy OLS F 检验）：

$$
F = \frac{(RSS_r - RSS_u)/p}{RSS_u/(n-2p-1)} \sim F(p, n-2p-1)
$$

其中 restricted 模型只用收益自身滞后，unrestricted 加入情感滞后；H0 为「情感不格兰杰导致收益」。

**本次结果（模拟 K 线，仅验证方法）**：

| 指标 | 值 |
| --- | --- |
| 对齐样本 | 49 个小时桶 |
| 最优 lag | **1 小时** |
| Pearson r | **0.7662** |
| p 值 | **< 0.0001** |
| Granger 最优 lag | 1 |
| Granger p | **< 0.0001** |
| ADF（情感） | 0.0229（5% 水平平稳；收益 0.0001） |

> **重要声明**：模拟价格在生成时使用了 `drift_t = 0.03 * sentiment_{t-1}`，因此显著性是「估计器能恢复已知结构」的健全性检查，**不能解读为真实市场中情感能预测 BTC**。真实研究必须做样本外回测、交易成本、混杂因子控制与多重检验校正。

---

### 维度六：工程化闭环（MLOps & API）

**FastAPI 端点**（`src/api_server.py`，自动生成 `/docs`）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 仓库/图谱/LLM 状态，Docker healthcheck 使用 |
| GET | `/entities/{entity}/sentiment-trend?interval=1h&hours=168` | 情感趋势 + 负面占比 + 汇总 |
| GET | `/entities/{entity}/events?negative_only=true` | 实体事件时间线 |
| GET | `/graph/multihop?query=提现&hops=2&negative_only=true` | GraphRAG 多跳查询 |
| GET | `/graph/stats` | 实体/事件/关系/负面事件统计 |
| GET | `/routing/stats` | 路由率、平均 SDI、实际/避免成本、Agent 分布 |
| GET | `/alerts?entity=币安` | 负面突增告警 |
| POST | `/analyze` | 在线调试：输入文本 → ABSA 四元组 + Tri-Agent 决策 |

**Docker Compose 服务**：

| 服务 | 镜像/构建 | 端口 | 作用 |
| --- | --- | --- | --- |
| `zookeeper` | `confluentinc/cp-zookeeper:7.6.1` | 2181 | Kafka 协调 |
| `kafka` | `confluentinc/cp-kafka:7.6.1` | 9092 / 29092 | `raw_comments` topic |
| `elasticsearch` | `elasticsearch:8.13.4` | 9200 | 评论、决策、告警索引 |
| `neo4j` | `neo4j:5.20-community` | 7474 / 7687 | 事件图谱 |
| `api` | 本地 `Dockerfile` | 8000 | FastAPI |
| `processor` | 本地 `Dockerfile` | — | Kafka 消费者 + 图谱写入 |
| `producer` | 本地 `Dockerfile`（profile `demo`） | — | 模拟评论生产者 |

**可观测性与韧性**：

- 所有容器带 healthcheck，`depends_on: condition: service_healthy` 保证启动顺序；
- 路由决策全量审计（JSONL + ES），可离线回放、调阈值、做成本归因；
- ES/Neo4j 不可达时自动降级到本地 JSONL / 内存图；
- LLM 超时/失败重试 2 次后降级为委员会加权共识；
- `processor` 支持 SIGINT/SIGTERM 优雅关闭，未提交的 offset 会在重启后重放。

---

## 4. 快速开始

### 4.1 本地（conda）

```bash
git clone git@github.com:DingBangBang/opinion-and-event-graph-system.git
cd opinion-and-event-graph-system

# 创建环境（已实测 python=3.10）
conda env create -f environment.yml
conda activate opinion-system

# 一键离线 demo：生成 544 条数据 → 流处理 → 建图 → 量化 → 5 张图
python scripts/seed_demo_data.py

# 测试
python -m pytest -q
# 22 passed
```

demo 产物：

| 文件 | 内容 |
| --- | --- |
| `data/processed.jsonl` | 544 条结构化评论（四元组 + Tri-Agent 决策 + 事件） |
| `data/routing_audit.jsonl` | 每次 SDI 路由的完整审计 |
| `data/alerts.jsonl` | 负面突增告警 |
| `data/graph_snapshot.json` | 内存图快照（可直接检查边属性） |
| `reports/demo_metrics.json` | 全部实验指标（README 中的数字来源） |
| `reports/*.png` | 5 张可视化结果 |

### 4.2 Docker Compose（完整闭环）

```bash
cp .env.example .env
# 可选：填入 DEEPSEEK_API_KEY；不填则 LLM 自动降级为委员会共识

docker compose up -d --build          # ZK + Kafka + ES + Neo4j + API + Processor
docker compose --profile demo up -d   # 再启动模拟生产者（1000 条，5 条/秒）

curl http://localhost:8000/health
docker compose logs -f processor      # 观察处理与告警
```

访问：

- FastAPI Swagger：<http://localhost:8000/docs>
- Neo4j Browser：<http://localhost:7474>（`neo4j / opinion-graph`，可用 `.env` 的 `NEO4J_PASSWORD` 修改）
- Elasticsearch：<http://localhost:9200>

启用真实 Transformer：

```bash
# 镜像体积 +~2GB；首次运行设置 ABSA_ALLOW_MODEL_DOWNLOAD=true
INSTALL_ML=true docker compose build
ABSA_ALLOW_MODEL_DOWNLOAD=true docker compose up -d
```

> 国内网络若遇到 pip SSL/超时：`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build`。

本地 Kafka 生产者（容器外）：

```bash
python scripts/mock_producer.py --bootstrap localhost:29092 --count 200 --rate 5
python -m src.stream_processor --from-beginning
```

### 4.3 关键环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SDI_THRESHOLD` | `0.45` | 路由阈值，越高越省钱、但复杂样本风险越大 |
| `ABSA_BACKEND` | `auto` | `auto` 优先 DeepSeek（有 Key）→ HF（有模型）→ lexicon |
| `DEEPSEEK_API_KEY` | 空 | 不填也能跑，LLM 走离线共识 |
| `ENABLE_FINBERT` | `true` | 无模型时自动用金融子词典 |
| `GRAPH_BACKEND` | `auto` | `neo4j` / `memory` / `auto` |
| `ALERT_WINDOW_SECONDS` | `300` | 负面突增窗口 |
| `ALERT_MIN_SAMPLES` | `8` | 窗口内最少样本数 |
| `ALERT_NEGATIVE_RATIO` | `0.6` | 负面占比阈值 |
| `ES_URL` / `NEO4J_URI` | localhost | Compose 内自动改为服务名 |

---

## 5. API 使用示例

```bash
# 1) 情感趋势
curl "http://localhost:8000/entities/币安/sentiment-trend?interval=1h&hours=72"

# 2) 负面事件时间线
curl "http://localhost:8000/entities/币安/events?negative_only=true&limit=20"

# 3) GraphRAG：提现相关的负面事件（支持按方面词而不是实体名查询）
curl "http://localhost:8000/graph/multihop?query=提现&hops=2&negative_only=true"

# 4) 路由成本
curl "http://localhost:8000/routing/stats"

# 5) 在线分析
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"币安提现三天没到账，客服还敷衍，太失望了","entity_hint":"币安"}'
```

`/graph/multihop` 返回结构：

```json
{
  "query_entity": "提现",
  "hops": 2,
  "negative_only": true,
  "events": [
    {
      "event_id": "evt_5c1f...",
      "label": "币安·提现·负面事件",
      "timestamp": "2026-10-20T12:58:00+00:00",
      "sentiment": -0.8854,
      "sentiment_label": "negative",
      "comment_id": "c_7a2...",
      "hops": 1
    }
  ],
  "paths": [["币安", "币安·提现·负面事件"]]
}
```

`/routing/stats` 返回：

```json
{
  "total": 544,
  "routed_to_llm": 110,
  "routing_rate": 0.2022,
  "avg_sdi": 0.2369,
  "estimated_cost_usd": 0.0,
  "avoided_cost_usd": 0.07103115,
  "by_agent": {"vader": 415, "llm": 110, "finbert": 19}
}
```

> `estimated_cost_usd=0` 是因为离线 demo 未配置 API Key、LLM 走 fallback；`avoided_cost_usd` 是被跳过调用的影子成本。配置 Key 后同一路由率下会看到真实 `estimated_cost_usd`。

---

## 6. 实验结果与可视化

### 6.1 实验设置

- 数据：`scripts/mock_producer.py` 生成 544 条评论，覆盖 11 个实体（7 家交易所 + BTC/ETH/SOL/USDT），72 小时；
- 场景权重：提现负面 18%、提现正面 14%、手续费 10%、客服 9%、App 6%、安全 6%、中性行情 10%、混合复杂 8%、合规 4%、通用 5%；
- 注入事件：在结束前 2 小时集中 24 条「币安提现失败」，每 6 秒一条，用于验证滑动窗口；
- 情感模型：离线 lexicon（保证可复现）；LLM 未配置，按 fallback 记录；
- 聚合：小时桶。

### 6.2 总量指标

| 指标 | 结果 |
| --- | --- |
| 生成 / 处理 | 544 / 544（0 error） |
| ABSA 四元组 | 1,720（平均 3.16 条/评论） |
| 负面评论占比 | 51.84% |
| Tri-Agent 平均耗时 | 0.125 ms（纯本地小模型） |
| LLM 路由率 | 20.22%（110 / 544） |
| Agent 分布 | VADER 415 / LLM 110 / FinBERT 19 |
| 平均 SDI | 0.2369 |
| 避免成本（影子） | $0.0710 / 544 条 ≈ $130 / 百万条 |
| 告警 | 1 次（币安提现负面 100%，8/8） |
| 图谱 | 11 实体 / 544 事件 / 1,497 关系 / 282 负面事件 |
| 量化 | lag=1，r=0.7662，Granger p<0.0001（模拟 K 线） |

### 6.3 图表

**① 实体情感趋势与负面占比**（`reports/sentiment_trend.png`）

![sentiment trend](reports/sentiment_trend.png)

**② ABSA 方面 × 极性分布**（`reports/absa_aspects.png`）——从 05 的「词云」升级为可运营的方面结构：

![absa aspects](reports/absa_aspects.png)

**③ Tri-Agent SDI 路由经济性**（`reports/routing_dashboard.png`）：左图阈值把 SDI 分布切成「小模型区 / LLM 区」，中图是 Agent 选择，右图是累计成本：

![routing dashboard](reports/routing_dashboard.png)

**④ GraphRAG 预览**（`reports/graph_preview.png`）：左侧实体、右侧事件，红线为负面事件：

![graph preview](reports/graph_preview.png)

**⑤ 情感-价格四联图**（`reports/sentiment_price_correlation.png`）：情感/价格、负面占比、滞后相关、散点与拟合：

![sentiment price](reports/sentiment_price_correlation.png)

### 6.4 关键结论

1. **ABSA 把「差评」拆成了可执行的问题单**：样本中「提现」相关四元组 236 条、「客服」188 条、「手续费」155 条，运营可以直接按方面排优先级；
2. **SDI 路由确实把预算花在刀刃上**：约 80% 的评论由小模型毫秒级处理；只有「VADER 与 FinBERT 明显冲突」的 20% 进入 LLM；
3. **图谱让舆情从「列表」变成「网络」**：`INVOLVES` 边同时带时间和极性，能回答「提现问题在哪些平台、什么时间、和哪些其他实体一起出现」；
4. **告警窗口有效**：24 条集中负面在窗口内触发一次告警，且冷却机制避免刷屏；
5. **量化流程可复现**：估计器在已知 lag=1 的模拟数据上恢复出 r≈0.76、Granger p<0.0001，说明数据管道和检验实现正确；真实市场结论需要 `--live` 与更严格的因果设计。

> 关于阈值的敏感性（同一批 544 条数据）：`0.35 → 41.4%` 路由、`0.45 → 20.2%`、`0.60 → 9.0%`。阈值不是越低越好：越低越安全但越贵，建议用标注集优化「路由后的 F1 / 成本」曲线。

---

## 7. 成本优势与 GraphRAG 业务价值

### 7.1 为什么 SDI 路由能省钱

如果每条评论都调用 LLM，成本随流量线性增长；但舆情数据是**长尾分布**：大多数评论情感明确、表达简单，真正需要跨句推理的是少数。

| 策略 | LLM 调用比例 | 百万条调用数 | 估算成本（DeepSeek 价） | 相对全量 |
| --- | --- | --- | --- | --- |
| 全量 LLM | 100% | 1,000,000 | ≈ **$160** | — |
| SDI θ=0.35 | 41.4% | 414,000 | ≈ **$66** | -58.6% |
| **SDI θ=0.45（默认）** | **20.2%** | **202,000** | **≈ $32** | **-79.8%** |
| SDI θ=0.60 | 9.0% | 90,000 | ≈ **$14** | -91.0% |

> 估算基于 `prompt≈220+len/1.5`、`completion≈90`、输入 $0.27/M、输出 $1.10/M；实际成本随文本长度、模型、缓存命中变化。`reports/routing_dashboard.png` 的右图给出同一批数据的累计曲线。

除了直接 token 成本，路由还带来：

- **延迟**：80% 请求在小模型上 0.1~20ms 完成，P95 不被 LLM 网络抖动拖累；
- **稳定性**：LLM 故障时只有 20% 流量受影响，且可降级为共识；
- **可解释**：每条决策保留 VADER/FinBERT/SDI/最终选择，便于审计和阈值调优；
- **可迭代**：换模型、改价格、改阈值都不需要改业务代码。

### 7.2 GraphRAG 的业务价值

传统「情感列表 + 关键词搜索」只能回答**单点问题**；图谱把评论投影成 `Entity–Event–Aspect` 网络后，可以回答**关系问题**：

| 业务问题 | 传统做法 | GraphRAG 做法 |
| --- | --- | --- |
| 「提现问题最近影响了哪些平台？」 | 关键词搜「提现」，人工翻几百条 | `multihop("提现", hops=2, negative_only=true)` 直接返回事件与路径 |
| 「币安的负面是否蔓延到 USDT？」 | 分别看两个实体的趋势，靠人判断 | 2 跳路径 `币安 ← 提现事件 → USDT`，按时间排序 |
| 「这次负面爆发的上游原因？」 | 无法关联方面与事件 | 事件的主方面 + 边上的 `opinion`（如「没到账」「手续费高」） |
| 「监管问询时如何自证？」 | 截图/Excel，不可追溯 | 每条边带 `comment_id`，可回查原文与模型决策 |
| 「给 LLM 的 RAG 上下文？」 | 向量检索相似句，缺结构 | 返回子图 + 时间线，LLM 可基于事实链回答 |

**一个真实可复现的查询**（离线 demo，`hops=2`，负面事件）返回的事件包括「币安·提现·负面事件」「Gate.io·提现·负面事件」「币安·手续费·负面事件」等，说明系统不仅能找到直接提及「提现」的评论，还能沿同一实体的其他事件扩展到关联风险。生产环境建议把 `hops` 限制在 2~3，避免图爆炸。

### 7.3 从 05 到现在的「意义」

- **分析范式**：从「文本 → 主题词」升级为「文本 → 结构化事实 → 关系网络 → 可查询服务」；
- **成本范式**：从「全量调用大模型」升级为「小模型优先 + 分歧时仲裁」，把大模型用在真正不确定的样本上；
- **工程范式**：从「Notebook/脚本」升级为「可编排、可观测、可降级、可测试」的实时系统；
- **业务范式**：从「事后报告」升级为「分钟级预警 + 根因定位 + 量化信号」。

---

## 8. 项目结构

```text
opinion-and-event-graph-system/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── environment.yml
├── requirements.txt / requirements-ml.txt
├── .env.example
├── config/
│   └── settings.py              # 统一配置（pydantic-settings）
├── src/
│   ├── absa_engine.py           # 维度一：ABSA（HF / DeepSeek / lexicon）
│   ├── tri_agent.py             # 维度二：VADER/FinBERT/LLM + SDI 路由 + 审计
│   ├── stream_processor.py      # 维度三：Kafka 消费 + 滑动窗口告警
│   ├── graph_builder.py         # 维度四：Neo4j / 内存图 + 多跳查询
│   ├── quant_signal.py          # 维度五：滞后相关 + 格兰杰检验
│   ├── api_server.py            # 维度六：FastAPI
│   ├── event_detector.py        # 评论 → 事件投影
│   ├── storage.py               # ES / 本地 JSONL 双实现
│   ├── schemas.py               # Pydantic 契约（Kafka/ES/API 共用）
│   ├── sentiment_rules.py       # 词典、否定、程度副词、统计检验辅助
│   ├── llm_client.py            # OpenAI 兼容客户端 + 重试 + JSON 解析
│   ├── logging_conf.py
│   └── resources/lexicons.py    # 实体别名 / 方面词 / 情感词典
├── scripts/
│   ├── mock_producer.py         # Kafka 模拟生产者（可复用数据生成器）
│   ├── seed_demo_data.py        # 离线端到端 demo + 5 张图
│   ├── bootstrap_storage.py     # ES 索引 + Neo4j 约束初始化
│   └── init_neo4j.cypher        # Schema/查询参考
├── notebooks/
│   └── sentiment_price_correlation.ipynb
├── tests/                       # 22 个测试
├── reports/                     # 已提交的可视化结果 + demo_metrics.json
├── data/                        # 运行时数据（大文件 gitignore，保留 sample）
└── legacy/
    └── 05_Comments_NLP/         # 原始项目代码（溯源，不参与运行）
```

---

## 9. 测试与质量

```bash
python -m pytest -q
# 22 passed
```

覆盖范围：

| 文件 | 覆盖点 |
| --- | --- |
| `test_absa_engine.py` | 中文四元组、实体提示回退、英文 ABSA、实体/方面识别、空文本 |
| `test_tri_agent.py` | SDI 公式、低分歧跳过 LLM、高分歧路由、阈值生效、审计落盘 |
| `test_graph_pipeline.py` | 处理→ES/JSONL→图谱、`INVOLVES` 边属性、多跳负面查询、事件幂等 |
| `test_quant_signal.py` | 已知 lag 恢复、Granger 显著性、样本不足安全返回 |
| `test_api.py` | health、趋势、事件、多跳、在线分析、路由/告警接口 |

另外通过：

- `python -m py_compile src/*.py scripts/*.py config/*.py`
- `docker compose config`（服务/环境变量/volume 语法校验）
- 离线端到端 demo：544 条、0 error、1 alert、5 张图

---

## 10. 已知限制与生产化建议

### 10.1 当前限制

1. **离线词典是「可复现的基线」而非 SOTA**：它能保证 CI/演示稳定，但真实中文 ABSA 建议启用 HF 或 DeepSeek；
2. **中文 VADER 是等价规则实现**：`vaderSentiment` 只支持英文，中文走同一套否定/程度副词规则，命名保持 VADER Agent 以体现「词级小模型」角色；
3. **实体表是金融/加密领域子集**：新平台需要扩充 `ENTITY_ALIASES`，或接入 NER；
4. **量化使用模拟 K 线**：结果只验证方法；`--live` 接入 Binance 后仍需做平稳性、协整、样本外回测；
5. **单节点基础设施**：Compose 面向演示/单机，生产需 Kafka 多 broker、ES 集群、Neo4j 集群、认证与 TLS；
6. **LLM JSON 稳定性**：已做 fence 提取、字段校验、失败降级，但极端输出仍可能触发 fallback；
7. **图谱规模**：当前一跳事件全量写入，百万级需要按时间分区、批量 UNWIND、异步写入；
8. **隐私与合规**：评论可能含个人信息，生产需脱敏、访问控制与审计保留策略。

### 10.2 Roadmap

- [ ] 引入主动学习：把 LLM 仲裁结果作为弱标签，蒸馏回小模型，进一步降低路由率；
- [ ] 方面级 GraphRAG：把 `Aspect` 也建模为节点，支持「手续费 → 提现 → 客服」的方面链；
- [ ] 向量检索 + 图检索混合：ES kNN + Neo4j 子图，作为 LLM 的 RAG 上下文；
- [ ] 实时看板：Grafana / Streamlit 展示实体热度、告警、成本；
- [ ] 模型服务化：Triton / vLLM，支持批量 GPU 推理；
- [ ] 数据质量：评论去重、机器人检测、水军聚类；
- [ ] CI/CD：GitHub Actions 跑 pytest + Docker build + compose smoke test。

---

## 11. 复现与引用

```bash
# 完整复现（无需 Docker / API Key）
conda env create -f environment.yml && conda activate opinion-system
python scripts/seed_demo_data.py
cat reports/demo_metrics.json
```

如果这个项目对你有帮助，欢迎 Star。原始 05 项目与爬虫代码保留在 [`legacy/05_Comments_NLP`](./legacy/05_Comments_NLP)，仅用于溯源；其中的 Cookie、代理是历史示例值，早已失效，请勿直接使用。

**License**：MIT（代码）；评论数据为脚本生成，不含真实用户信息。

---

## 12. 一句话总结

> 05 解决了「用户说了什么」，这个系统进一步回答「**谁、在哪个方面、什么时间、和谁一起、有多负面、值不值得调用大模型**」——并用可编排的流处理、可追溯的图谱和可复现的实验，把答案变成能上线的服务。



