"""Mock real-time comment producer for the ``raw_comments`` Kafka topic.

Also exposes :func:`generate_comments`, reused by ``scripts/seed_demo_data.py``
so the offline demo and the live pipeline share identical data generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas import RawComment  # noqa: E402

EXCHANGES = ["币安", "OKX", "火币HTX", "Coinbase", "Bybit", "Bitget", "Gate.io"]
ASSETS = ["BTC", "ETH", "SOL", "USDT"]

NEGATIVE_WITHDRAWAL = (
    "在{entity}提现{amount}U，已经{delay}小时了还没到账，客服只会复制粘贴，太失望了。",
    "救命，{entity}提现一直失败，页面显示处理中，资金到底去哪了？",
    "{entity}提现不到账还扣手续费，问客服就让我等，真的很垃圾。",
    "昨天在{entity}提币，到现在没到账，冻结我的钱，准备投诉。",
)
POSITIVE_WITHDRAWAL = (
    "{entity}提现秒到，手续费也低，App很流畅，体验不错，推荐。",
    "刚在{entity}试了一下提现，几分钟就到账了，安全稳定，很放心。",
    "{entity}的出金速度真的快，低手续费，客服也耐心，好评。",
)
NEGATIVE_FEE = (
    "{entity}手续费高得离谱，充值还要额外收费，感觉被坑了。",
    "{entity}的手续费越来越贵，滑点也大，小散户根本没法玩。",
    "{entity}乱收费，提现要收{amount}U手续费，太坑人了。",
)
NEGATIVE_SECURITY = (
    "{entity}账号被盗，客服拒绝处理，资金被冻结，准备报警。",
    "在{entity}被钓鱼网站骗了，平台推诿不处理，非常不安全。",
    "{entity}风控乱封号，账户里的钱取不出来，黑平台吧。",
)
NEGATIVE_SERVICE = (
    "{entity}客服态度太差了，一直在敷衍我，真的很失望。",
    "联系{entity}人工客服排队一小时，最后机器人回复，太离谱了。",
    "{entity}客服推诿扯皮，问题拖了三天都没解决，体验糟糕。",
)
NEGATIVE_APP = (
    "{entity}的App一直闪退，登录不上，页面卡死，难用死了。",
    "{entity}客户端更新后疯狂卡顿，行情延迟，体验极差。",
    "{entity} App崩了，下单一直失败，客服也不管，气死了。",
)
POSITIVE_SERVICE = (
    "{entity}客服很耐心，问题很快解决，体验不错。",
    "{entity}客服响应及时，处理专业，值得信任。",
    "{entity}的App很流畅，界面清晰，使用方便，满意。",
)
NEUTRAL_MARKET = (
    "今天{asset}行情震荡，K线走得比较纠结，观望为主。",
    "{asset}价格在区间内波动，{entity}的深度看起来还算正常。",
    "{asset}成交量一般，市场情绪中性，暂时没有明显方向。",
)
MIXED = (
    "虽然{entity}提现速度还行，但手续费越来越贵，客服又敷衍，整体体验很差。",
    "{entity}到账挺快的，可是App经常闪退，安全性也让人担心。",
    "不得不说{entity}的客服态度不错，但提现审核太久了，还是有点失望。",
    "{entity}手续费低是优点，不过行情剧烈时老是宕机，稳定性不行。",
)
COMPLIANCE = (
    "{entity}没有合规牌照，监管风险太大，不敢继续用了。",
    "{entity}最近被监管点名，提现通道受限，大家小心。",
)
GENERIC = (
    "{entity}整体体验一般般，没什么特别的，凑合用吧。",
    "{entity}用了一段时间，感觉还行，希望继续改进。",
)


def _comment_id(entity: str, index: int, when: datetime, rng: random.Random) -> str:
    raw = f"{entity}:{index}:{when.isoformat()}:{rng.random()}".encode()
    return "c_" + hashlib.sha1(raw).hexdigest()[:16]


def _pick(seq: tuple[str, ...], rng: random.Random) -> str:
    return rng.choice(seq)


def _render(template: str, entity: str, asset: str, rng: random.Random) -> str:
    return template.format(
        entity=entity,
        asset=asset,
        amount=rng.choice([100, 500, 1000, 5000, 10000, 50000]),
        delay=rng.choice([2, 6, 12, 24, 48, 72]),
        pct=rng.choice([1, 3, 5, 8, 12]),
    )


def generate_comment(
    index: int,
    when: datetime | None = None,
    seed: int | None = None,
    force: str | None = None,
    entity_override: str | None = None,
) -> RawComment:
    """Generate one bilingual/mixed comment with a consistent entity hint."""

    rng = random.Random(seed if seed is not None else index)
    when = when or datetime.now(timezone.utc)
    entity = entity_override or rng.choice(EXCHANGES)
    asset = rng.choice(ASSETS)

    scenario = force or rng.choices(
        [
            "withdrawal_neg",
            "withdrawal_pos",
            "fee_neg",
            "security_neg",
            "service_neg",
            "app_neg",
            "service_pos",
            "market_neutral",
            "mixed",
            "compliance_neg",
            "generic",
        ],
        weights=[18, 14, 10, 6, 9, 6, 10, 10, 8, 4, 5],
        k=1,
    )[0]

    if scenario == "withdrawal_neg":
        text = _render(_pick(NEGATIVE_WITHDRAWAL, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "withdrawal_pos":
        text = _render(_pick(POSITIVE_WITHDRAWAL, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "fee_neg":
        text = _render(_pick(NEGATIVE_FEE, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "security_neg":
        text = _render(_pick(NEGATIVE_SECURITY, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "service_neg":
        text = _render(_pick(NEGATIVE_SERVICE, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "app_neg":
        text = _render(_pick(NEGATIVE_APP, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "service_pos":
        text = _render(_pick(POSITIVE_SERVICE, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "market_neutral":
        text = _render(_pick(NEUTRAL_MARKET, rng), entity, asset, rng)
        entity_hint = asset if rng.random() < 0.5 else entity
    elif scenario == "mixed":
        text = _render(_pick(MIXED, rng), entity, asset, rng)
        entity_hint = entity
    elif scenario == "compliance_neg":
        text = _render(_pick(COMPLIANCE, rng), entity, asset, rng)
        entity_hint = entity
    else:
        text = _render(_pick(GENERIC, rng), entity, asset, rng)
        entity_hint = entity

    return RawComment(
        comment_id=_comment_id(entity, index, when, rng),
        text=text,
        source="mock_producer",
        entity_hint=entity_hint,
        author_id=f"user_{rng.randint(1000, 9999)}",
        created_at=when,
        metadata={"scenario": scenario, "asset": asset},
    )


def generate_comments(
    count: int = 400,
    hours: int = 72,
    seed: int = 42,
    end: datetime | None = None,
    burst: bool = True,
) -> list[RawComment]:
    """Generate a time-ordered dataset with one deliberate negative spike.

    The burst makes the sliding-window detector fire during the offline demo and
    also creates a visible sentiment dip for the quant notebook.
    """

    end = end or datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    rng = random.Random(seed)
    comments: list[RawComment] = []
    span = (end - start).total_seconds()

    for index in range(count):
        when = start + timedelta(seconds=rng.uniform(0, span))
        comments.append(generate_comment(index, when, seed=seed + index))

    if burst:
        burst_at = end - timedelta(hours=2)
        for offset in range(24):
            when = burst_at + timedelta(seconds=offset * 6)
            comments.append(
                generate_comment(
                    count + offset,
                    when,
                    seed=seed + 10_000 + offset,
                    force="withdrawal_neg",
                    entity_override="币安",
                )
            )

    comments.sort(key=lambda item: item.created_at)
    return comments


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Kafka raw_comments mock producer")
    parser.add_argument("--bootstrap", default="localhost:29092")
    parser.add_argument("--topic", default="raw_comments")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--rate", type=float, default=5.0, help="messages per second")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jsonl", type=str, default=None, help="also/only write JSONL")
    parser.add_argument("--dry-run", action="store_true", help="print instead of sending")
    parser.add_argument("--staggered", action="store_true", help="timestamps spread over 1h")
    args = parser.parse_args()

    producer = None
    if not args.dry_run and not args.jsonl:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap.split(","),
            value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            linger_ms=50,
            retries=3,
        )

    handle = Path(args.jsonl).open("w", encoding="utf-8") if args.jsonl else None
    now = datetime.now(timezone.utc)
    sent = 0
    try:
        for index in range(args.count):
            when = now - timedelta(seconds=(args.count - index) * 0.2) if args.staggered else None
            comment = generate_comment(index, when=when, seed=args.seed)
            payload = comment.model_dump(mode="json")
            if producer is not None:
                producer.send(args.topic, payload)
            if handle is not None:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if args.dry_run:
                print(json.dumps(payload, ensure_ascii=False))
            sent += 1
            if sent % 20 == 0:
                print(f"… sent {sent}/{args.count}")
            if producer is not None and args.rate > 0:
                time.sleep(1.0 / args.rate)
        if producer is not None:
            producer.flush()
    finally:
        if producer is not None:
            producer.close()
        if handle is not None:
            handle.close()
    print(f"✓ producer finished: {sent} messages on topic '{args.topic}'")


if __name__ == "__main__":  # pragma: no cover
    main()

