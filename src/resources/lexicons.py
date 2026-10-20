"""Bilingual domain lexicons used by the offline ABSA / VADER / FinBERT fallbacks.

The lexicons are explicit so every number in the demo is reproducible without
downloading a model or calling an API. When a Transformer or LLM backend is
enabled the same taxonomy (entity types / aspect categories) is reused, so the
graph schema stays stable across backends.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Entity aliases: canonical name -> (aliases, entity type)
# Longer aliases are matched first to avoid partial hits.
# ---------------------------------------------------------------------------
ENTITY_ALIASES: Final[dict[str, tuple[tuple[str, ...], str]]] = {
    "币安": (("币安", "binance", "币安app", "binance app"), "exchange"),
    "OKX": (("okx", "欧易", "okex"), "exchange"),
    "火币HTX": (("火币", "htx", "huobi"), "exchange"),
    "Coinbase": (("coinbase",), "exchange"),
    "Bybit": (("bybit",), "exchange"),
    "Bitget": (("bitget",), "exchange"),
    "Gate.io": (("gate.io", "gateio", "芝麻开门"), "exchange"),
    "Kraken": (("kraken",), "exchange"),
    "老虎证券": (("老虎证券", "tiger brokers", "tiger trade"), "broker"),
    "富途": (("富途", "futu", "moomoo"), "broker"),
    "支付宝": (("支付宝", "alipay"), "payment"),
    "微信支付": (("微信支付", "wechat pay"), "payment"),
    "BTC": (("btc", "比特币", "bitcoin"), "asset"),
    "ETH": (("eth", "以太坊", "ethereum"), "asset"),
    "USDT": (("usdt", "泰达币", "tether"), "asset"),
    "SOL": (("sol", "solana"), "asset"),
    "XRP": (("xrp", "瑞波币", "ripple"), "asset"),
    "DOGE": (("doge", "狗狗币", "dogecoin"), "asset"),
    "银行": (("银行", "bank"), "institution"),
}

# ---------------------------------------------------------------------------
# Aspect taxonomy: category -> (Chinese label, keywords)
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "withdrawal": ("提现", ("提现", "取现", "出金", "withdraw", "withdrawal", "提币", "转出")),
    "deposit": ("充值", ("充值", "入金", "充币", "deposit", "存币")),
    "fee": ("手续费", ("手续费", "费率", "佣金", "gas费", "fee", "fees", "滑点")),
    "security": ("安全性", ("安全", "被盗", "盗号", "钓鱼", "诈骗", "骗子", "资金安全", "security", "scam", "hack")),
    "customer_service": ("客服", ("客服", "人工客服", "售后", "客服响应", "customer service", "support")),
    "app_experience": ("App体验", ("app", "客户端", "页面", "界面", "卡顿", "闪退", "崩溃", "登录", "体验", "ui", "lag")),
    "settlement_speed": ("到账速度", ("到账", "速度", "多久", "延迟", "秒到", "处理时间", "settlement", "delay")),
    "kyc": ("实名认证", ("实名", "认证", "kyc", "身份验证", "审核", "风控")),
    "market": ("行情", ("行情", "价格", "币价", "涨", "跌", "k线", "market", "price")),
    "liquidity": ("流动性", ("流动性", "深度", "成交", "买卖盘", "liquidity", "slippage")),
    "product": ("产品功能", ("产品", "功能", "合约", "杠杆", "理财", "staking", "feature", "product")),
    "compliance": ("合规", ("合规", "监管", "牌照", "法规", "compliance", "regulation")),
    "general": ("整体体验", ("平台", "服务", "公司", "整体", "体验", "platform", "service")),
}


# ---------------------------------------------------------------------------
# Word-level sentiment lexicon (VADER-style negation / intensifier rules apply)
# ---------------------------------------------------------------------------
POSITIVE_TERMS: Final[dict[str, float]] = {
    "到账快": 0.9, "秒到": 1.0, "提现快": 0.9, "很快": 0.6, "快速": 0.55,
    "稳定": 0.6, "安全": 0.7, "靠谱": 0.8, "专业": 0.6, "好用": 0.7,
    "流畅": 0.65, "方便": 0.5, "便宜": 0.6, "低手续费": 0.8, "手续费低": 0.8,
    "良心": 0.85, "满意": 0.75, "喜欢": 0.6, "推荐": 0.6, "信任": 0.7,
    "放心": 0.7, "给力": 0.8, "优秀": 0.8, "强大": 0.6, "清晰": 0.5,
    "及时": 0.55, "耐心": 0.6, "解决": 0.5, "成功": 0.5, "顺利": 0.6,
    "高效": 0.7, "透明": 0.6, "实惠": 0.6, "不错": 0.6, "棒": 0.7,
    "赞": 0.7, "牛": 0.6, "稳": 0.5, "赚": 0.5, "盈利": 0.6,
    "省心": 0.65, "贴心": 0.7, "划算": 0.6,
    "fast": 0.6, "quick": 0.6, "great": 0.7, "excellent": 0.85,
    "smooth": 0.65, "reliable": 0.7, "recommend": 0.6, "good": 0.5,
    "happy": 0.65, "secure": 0.7,
}

NEGATIVE_TERMS: Final[dict[str, float]] = {
    "提现失败": 0.95, "无法提现": 0.95, "提现不了": 0.95, "不到账": 0.9,
    "没到账": 0.85, "延迟": 0.5, "卡顿": 0.65, "崩溃": 0.85, "闪退": 0.8,
    "卡死": 0.85, "诈骗": 1.0, "骗子": 1.0, "跑路": 1.0, "黑平台": 1.0,
    "割韭菜": 0.9, "亏损": 0.75, "暴跌": 0.8, "被盗": 0.95, "盗号": 0.9,
    "冻结": 0.9, "封号": 0.9, "风控": 0.5, "霸王条款": 0.9, "高额手续费": 0.8,
    "手续费高": 0.8, "太贵": 0.7, "坑": 0.7, "坑人": 0.85, "恶心": 0.9,
    "垃圾": 0.9, "差劲": 0.8, "失望": 0.75, "生气": 0.8, "愤怒": 0.85,
    "投诉": 0.6, "无语": 0.6, "离谱": 0.8, "糟糕": 0.8, "失败": 0.7,
    "错误": 0.55, "问题": 0.4, "麻烦": 0.45, "慢": 0.5, "难用": 0.7,
    "复杂": 0.4, "不稳定": 0.7, "不安全": 0.85, "泄露": 0.9, "隐瞒": 0.7,
    "推诿": 0.75, "敷衍": 0.75, "机器人": 0.5, "排队": 0.4, "审核久": 0.7,
    "认证失败": 0.8, "滑点": 0.5, "拔网线": 0.9, "宕机": 0.9,
    "跑不出去": 0.9, "爆仓": 0.85, "亏": 0.6, "差": 0.55, "烂": 0.85,
    "难受": 0.6, "着急": 0.55, "拖": 0.5, "拒绝": 0.65, "不退": 0.8,
    "虚假": 0.8, "诱导": 0.7, "骚扰": 0.7, "捆绑": 0.6, "乱收费": 0.85,
    "delayed": 0.7, "delay": 0.5, "slow": 0.5, "terrible": 0.9,
    "awful": 0.85, "worst": 0.9, "bad": 0.6, "poor": 0.65,
    "broken": 0.75, "crash": 0.8, "freeze": 0.8, "scam": 0.95, "fraud": 0.95,
}

NEGATIONS: Final[tuple[str, ...]] = (
    "并不", "不是", "不太", "不够", "没有", "没能", "毫无", "无法", "不能",
    "不会", "不再", "不", "没", "未", "别", "无", "非",
    "not", "no", "never", "without", "cannot", "can't", "didn't", "isn't",
    "wasn't", "don't", "doesn't",
)

INTENSIFIERS: Final[dict[str, float]] = {
    "非常": 1.6, "特别": 1.5, "很": 1.3, "太": 1.4, "极其": 1.8, "超级": 1.7,
    "十分": 1.5, "相当": 1.4, "真的": 1.3, "巨": 1.5, "最": 1.5, "更": 1.2,
    "越来越": 1.4, "严重": 1.6, "完全": 1.4, "根本": 1.4, "一直": 1.2,
    "very": 1.4, "extremely": 1.8, "really": 1.3, "so": 1.2, "too": 1.3,
    "highly": 1.5, "absolutely": 1.6, "totally": 1.4,
}

# Words ignored when guessing an opinion term next to an aspect.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "的", "了", "是", "在", "我", "也", "都", "和", "就", "还", "又", "很",
        "太", "真", "这个", "那个", "平台", "app", "感觉", "觉得", "使用",
        "the", "a", "an", "is", "was", "were", "it", "this", "that", "and",
        "to", "of", "for", "on", "in", "my", "i",
    }
)

# Sentence-level finance lexicon for the offline FinBERT surrogate.
# Only financially salient terms are included: emotional but non-financial words
# (e.g. 敷衍/失望) are deliberately missed, which is what creates realistic
# disagreement between VADER and FinBERT in the offline demo.
FINANCE_ZH_POSITIVE: Final[tuple[str, ...]] = (
    "到账快", "秒到", "提现快", "低手续费", "手续费低", "稳定", "安全",
    "盈利", "赚", "透明", "高效", "靠谱",
)
FINANCE_ZH_NEGATIVE: Final[tuple[str, ...]] = (
    "提现失败", "无法提现", "提现不了", "不到账", "没到账", "高额手续费",
    "手续费高", "被盗", "盗号", "冻结", "爆仓", "亏损", "诈骗", "骗子",
    "跑路", "黑平台", "滑点", "宕机", "封号", "乱收费",
)

FINANCE_POSITIVE: Final[tuple[str, ...]] = (
    "bullish", "beat", "growth", "profit", "secure", "stable", "upgrade",
    "strong", "gain", "approval", "partnership", "adoption",
)
FINANCE_NEGATIVE: Final[tuple[str, ...]] = (
    "bearish", "miss", "loss", "hack", "exploit", "lawsuit", "fraud",
    "outage", "delist", "ban", "freeze", "downgrade", "weak", "decline",
)

