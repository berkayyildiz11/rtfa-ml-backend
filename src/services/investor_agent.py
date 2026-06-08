from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from uuid import uuid4


INITIAL_CASH = 10_000.0
RUN_DAYS = 8
MIN_BUY_SCORE = 0.20
MIN_BUY_CONFIDENCE = 0.55
MAX_OPEN_POSITIONS = 5
MAX_POSITION_FRACTION = 0.25
DAILY_DEPLOY_CASH_FRACTION = 0.40
MIN_TRADE_AMOUNT = 50.0

PredictionBundleBuilder = Callable[[str], Awaitable[dict[str, dict[str, object]]]]


def agent_collections(db) -> dict[str, object]:
    return {
        "runs": db.agent_runs,
        "decisions": db.agent_decisions,
        "trades": db.agent_trades,
        "snapshots": db.agent_portfolio_snapshots,
    }


async def start_agent_run(
    db,
    *,
    initial_cash: float = INITIAL_CASH,
    duration_days: int = RUN_DAYS,
    now: datetime | None = None,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    existing = await db.agent_runs.find_one({"status": "active"}, {"_id": 0})
    if existing:
        return {
            "status": "already_active",
            "run": existing,
        }

    run = {
        "run_id": str(uuid4()),
        "status": "active",
        "started_at": now,
        "ends_at": now + timedelta(days=duration_days),
        "initial_cash": round(float(initial_cash), 2),
        "cash": round(float(initial_cash), 2),
        "positions": {},
        "duration_days": duration_days,
        "settings": {
            "periods": ["1d", "1w"],
            "score_weights": {"1d": 0.40, "1w": 0.60},
            "min_buy_score": MIN_BUY_SCORE,
            "min_buy_confidence": MIN_BUY_CONFIDENCE,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_position_fraction": MAX_POSITION_FRACTION,
            "daily_deploy_cash_fraction": DAILY_DEPLOY_CASH_FRACTION,
            "min_trade_amount": MIN_TRADE_AMOUNT,
        },
        "created_at": now,
        "updated_at": now,
    }
    await db.agent_runs.insert_one(run)
    run.pop("_id", None)
    return {
        "status": "started",
        "run": run,
    }


async def get_agent_status(db) -> dict[str, object]:
    run = await db.agent_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        run = await db.agent_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    if not run:
        return {"status": "not_started"}

    snapshot = await db.agent_portfolio_snapshots.find_one(
        {"run_id": run["run_id"]},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    return {
        "status": run["status"],
        "run": run,
        "latest_snapshot": snapshot,
    }


async def get_agent_history(db, *, limit: int = 100) -> dict[str, object]:
    run = await db.agent_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    if not run:
        return {"status": "not_started", "run": None, "decisions": [], "trades": [], "snapshots": []}

    run_id = run["run_id"]
    decisions = await db.agent_decisions.find(
        {"run_id": run_id},
        {"_id": 0},
    ).sort("created_at", -1).limit(limit).to_list(length=limit)
    trades = await db.agent_trades.find(
        {"run_id": run_id},
        {"_id": 0},
    ).sort("created_at", -1).limit(limit).to_list(length=limit)
    snapshots = await db.agent_portfolio_snapshots.find(
        {"run_id": run_id},
        {"_id": 0},
    ).sort("created_at", -1).limit(limit).to_list(length=limit)
    return {
        "status": "success",
        "run": run,
        "decisions": decisions,
        "trades": trades,
        "snapshots": snapshots,
    }


async def run_daily_agent_cycle(
    db,
    *,
    tickers: list[str],
    build_prediction_bundle: PredictionBundleBuilder,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    run = await db.agent_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        return {"status": "error", "message": "No active agent run found."}

    if now >= run["ends_at"]:
        return await liquidate_agent_run(
            db,
            price_lookup=_latest_prediction_price_lookup(tickers, build_prediction_bundle),
            now=now,
            reason="scheduled_end",
        )

    cycle_date = now.date().isoformat()
    existing_snapshot = await db.agent_portfolio_snapshots.find_one(
        {"run_id": run["run_id"], "cycle_date": cycle_date},
        {"_id": 0},
    )
    if existing_snapshot and not force:
        return {
            "status": "already_ran_today",
            "snapshot": existing_snapshot,
        }

    prediction_bundles: dict[str, dict[str, dict[str, object]]] = {}
    opportunities = []
    decisions = []
    for ticker in tickers:
        try:
            bundle = await build_prediction_bundle(ticker)
            prediction_bundles[ticker] = bundle
            opportunity = build_investment_opportunity(ticker, bundle["1d"], bundle["1w"])
            opportunities.append(opportunity)
        except Exception as exc:
            opportunity = {
                "ticker": ticker,
                "eligible": False,
                "action": "skip",
                "reason": f"prediction_error: {exc}",
                "combined_score": 0.0,
                "combined_confidence": 0.0,
                "investment_score": 0.0,
                "latest_price": None,
                "predictions": {},
            }
            opportunities.append(opportunity)

    run = await db.agent_runs.find_one({"run_id": run["run_id"]}, {"_id": 0})
    cash = float(run["cash"])
    positions = dict(run.get("positions", {}))
    initial_cash = float(run["initial_cash"])
    trades = []

    for opportunity in opportunities:
        ticker = opportunity["ticker"]
        position = positions.get(ticker)
        if not position:
            continue
        price = opportunity.get("latest_price")
        if price is None:
            continue
        if float(opportunity["combined_score"]) <= 0.0:
            trade = sell_position(
                run_id=run["run_id"],
                ticker=ticker,
                position=position,
                price=float(price),
                now=now,
                reason="model_score_not_positive",
            )
            cash += trade["cash_delta"]
            positions.pop(ticker, None)
            trades.append(trade)

    buy_candidates = select_buy_candidates(opportunities, positions)
    deployable_cash = min(cash, cash * DAILY_DEPLOY_CASH_FRACTION)
    buy_plan = allocate_cash(
        buy_candidates,
        available_cash=deployable_cash,
        portfolio_start_value=initial_cash,
        positions=positions,
    )

    for ticker, buy_amount in buy_plan.items():
        opportunity = next(item for item in buy_candidates if item["ticker"] == ticker)
        price = float(opportunity["latest_price"])
        quantity = buy_amount / price
        trade = buy_position(
            run_id=run["run_id"],
            ticker=ticker,
            amount=buy_amount,
            quantity=quantity,
            price=price,
            now=now,
            reason="high_confidence_positive_model_signal",
        )
        cash -= buy_amount
        positions[ticker] = merge_position(positions.get(ticker), trade)
        trades.append(trade)

    for opportunity in opportunities:
        decision = {
            "run_id": run["run_id"],
            "cycle_date": cycle_date,
            "ticker": opportunity["ticker"],
            "action": action_for_opportunity(opportunity, trades),
            "reason": reason_for_opportunity(opportunity, trades),
            "combined_score": opportunity["combined_score"],
            "combined_confidence": opportunity["combined_confidence"],
            "investment_score": opportunity["investment_score"],
            "latest_price": opportunity["latest_price"],
            "predictions": opportunity["predictions"],
            "created_at": now,
        }
        decisions.append(decision)

    if decisions:
        await db.agent_decisions.insert_many(decisions)
    if trades:
        await db.agent_trades.insert_many(trades)

    snapshot = build_portfolio_snapshot(
        run_id=run["run_id"],
        cycle_date=cycle_date,
        cash=cash,
        positions=positions,
        opportunities=opportunities,
        now=now,
    )
    await db.agent_portfolio_snapshots.insert_one(snapshot)
    await db.agent_runs.update_one(
        {"run_id": run["run_id"]},
        {
            "$set": {
                "cash": round(cash, 2),
                "positions": positions,
                "updated_at": now,
            }
        },
    )
    for item in [*decisions, *trades, snapshot]:
        item.pop("_id", None)

    return {
        "status": "success",
        "run_id": run["run_id"],
        "cycle_date": cycle_date,
        "decisions": decisions,
        "trades": trades,
        "snapshot": snapshot,
    }


async def liquidate_agent_run(
    db,
    *,
    price_lookup: Callable[[str], Awaitable[float | None]],
    now: datetime | None = None,
    reason: str = "manual_liquidation",
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    run = await db.agent_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        return {"status": "error", "message": "No active agent run found."}

    cash = float(run["cash"])
    positions = dict(run.get("positions", {}))
    trades = []
    for ticker, position in list(positions.items()):
        price = await price_lookup(ticker)
        if price is None:
            price = float(position.get("last_price", position["avg_cost"]))
        trade = sell_position(
            run_id=run["run_id"],
            ticker=ticker,
            position=position,
            price=float(price),
            now=now,
            reason=reason,
        )
        cash += trade["cash_delta"]
        positions.pop(ticker, None)
        trades.append(trade)

    if trades:
        await db.agent_trades.insert_many(trades)

    final_value = round(cash, 2)
    profit_loss = round(final_value - float(run["initial_cash"]), 2)
    return_pct = round(profit_loss / float(run["initial_cash"]) * 100, 4)
    snapshot = {
        "run_id": run["run_id"],
        "cycle_date": now.date().isoformat(),
        "cash": final_value,
        "positions": {},
        "market_value": 0.0,
        "total_equity": final_value,
        "profit_loss": profit_loss,
        "return_pct": return_pct,
        "created_at": now,
        "reason": reason,
    }
    await db.agent_portfolio_snapshots.insert_one(snapshot)
    await db.agent_runs.update_one(
        {"run_id": run["run_id"]},
        {
            "$set": {
                "status": "completed",
                "cash": final_value,
                "positions": {},
                "ended_at": now,
                "final_value": final_value,
                "profit_loss": profit_loss,
                "return_pct": return_pct,
                "updated_at": now,
            }
        },
    )
    for item in [*trades, snapshot]:
        item.pop("_id", None)

    return {
        "status": "completed",
        "run_id": run["run_id"],
        "trades": trades,
        "snapshot": snapshot,
    }


def build_investment_opportunity(
    ticker: str,
    prediction_1d: dict[str, object],
    prediction_1w: dict[str, object],
) -> dict[str, object]:
    score_1d = float(prediction_1d["prediction"]["score"])
    score_1w = float(prediction_1w["prediction"]["score"])
    confidence_1d = _prediction_confidence_score(prediction_1d)
    confidence_1w = _prediction_confidence_score(prediction_1w)
    combined_score = round(score_1d * 0.40 + score_1w * 0.60, 4)
    combined_confidence = round(confidence_1d * 0.40 + confidence_1w * 0.60, 4)
    investment_score = round(max(0.0, combined_score) * combined_confidence, 4)
    latest_price = float(prediction_1d.get("latest_price") or prediction_1w.get("latest_price"))
    eligible = combined_score > MIN_BUY_SCORE and combined_confidence >= MIN_BUY_CONFIDENCE

    return {
        "ticker": ticker.upper(),
        "eligible": eligible,
        "combined_score": combined_score,
        "combined_confidence": combined_confidence,
        "investment_score": investment_score,
        "latest_price": latest_price,
        "reason": "eligible" if eligible else "signal_or_confidence_too_weak",
        "predictions": {
            "1d": compact_prediction(prediction_1d),
            "1w": compact_prediction(prediction_1w),
        },
    }


def compact_prediction(prediction: dict[str, object]) -> dict[str, object]:
    return {
        "period": prediction["period"],
        "signal": prediction["prediction"]["signal"],
        "score": prediction["prediction"]["score"],
        "confidence": prediction["prediction"]["confidence"],
        "latest_price": prediction["latest_price"],
        "latest_price_time": prediction["latest_price_time"],
        "weights": prediction.get("weights", {}),
        "signals": prediction.get("signals", {}),
    }


def select_buy_candidates(
    opportunities: list[dict[str, object]],
    positions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    held = set(positions)
    candidates = [
        opportunity
        for opportunity in opportunities
        if opportunity["eligible"] and opportunity["ticker"] not in held
    ]
    candidates.sort(key=lambda item: item["investment_score"], reverse=True)
    remaining_slots = max(0, MAX_OPEN_POSITIONS - len(held))
    return candidates[:remaining_slots]


def allocate_cash(
    candidates: list[dict[str, object]],
    *,
    available_cash: float,
    portfolio_start_value: float,
    positions: dict[str, dict[str, object]],
) -> dict[str, float]:
    if not candidates or available_cash < MIN_TRADE_AMOUNT:
        return {}

    total_score = sum(float(candidate["investment_score"]) for candidate in candidates)
    if total_score <= 0:
        return {}

    allocation: dict[str, float] = {}
    max_position_value = portfolio_start_value * MAX_POSITION_FRACTION
    for candidate in candidates:
        ticker = candidate["ticker"]
        current_value = float(positions.get(ticker, {}).get("market_value", 0.0))
        cap_room = max(0.0, max_position_value - current_value)
        amount = available_cash * float(candidate["investment_score"]) / total_score
        amount = min(amount, cap_room)
        if amount >= MIN_TRADE_AMOUNT:
            allocation[ticker] = round(amount, 2)

    return allocation


def buy_position(
    *,
    run_id: str,
    ticker: str,
    amount: float,
    quantity: float,
    price: float,
    now: datetime,
    reason: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "ticker": ticker,
        "side": "buy",
        "quantity": round(quantity, 8),
        "price": round(price, 4),
        "cash_delta": round(-amount, 2),
        "notional": round(amount, 2),
        "reason": reason,
        "created_at": now,
    }


def sell_position(
    *,
    run_id: str,
    ticker: str,
    position: dict[str, object],
    price: float,
    now: datetime,
    reason: str,
) -> dict[str, object]:
    quantity = float(position["quantity"])
    notional = quantity * price
    cost_basis = quantity * float(position["avg_cost"])
    return {
        "run_id": run_id,
        "ticker": ticker,
        "side": "sell",
        "quantity": round(quantity, 8),
        "price": round(price, 4),
        "cash_delta": round(notional, 2),
        "notional": round(notional, 2),
        "realized_profit_loss": round(notional - cost_basis, 2),
        "reason": reason,
        "created_at": now,
    }


def merge_position(
    existing: dict[str, object] | None,
    trade: dict[str, object],
) -> dict[str, object]:
    quantity = float(trade["quantity"])
    cost = float(trade["notional"])
    if existing:
        quantity += float(existing["quantity"])
        cost += float(existing["cost_basis"])

    avg_cost = cost / quantity
    return {
        "ticker": trade["ticker"],
        "quantity": round(quantity, 8),
        "avg_cost": round(avg_cost, 4),
        "cost_basis": round(cost, 2),
        "last_price": trade["price"],
        "market_value": round(quantity * float(trade["price"]), 2),
        "opened_at": existing.get("opened_at") if existing else trade["created_at"],
        "updated_at": trade["created_at"],
    }


def build_portfolio_snapshot(
    *,
    run_id: str,
    cycle_date: str,
    cash: float,
    positions: dict[str, dict[str, object]],
    opportunities: list[dict[str, object]],
    now: datetime,
) -> dict[str, object]:
    prices = {
        opportunity["ticker"]: opportunity["latest_price"]
        for opportunity in opportunities
        if opportunity.get("latest_price") is not None
    }
    market_value = 0.0
    enriched_positions = {}
    for ticker, position in positions.items():
        price = float(prices.get(ticker) or position.get("last_price") or position["avg_cost"])
        quantity = float(position["quantity"])
        market_position = dict(position)
        market_position["last_price"] = round(price, 4)
        market_position["market_value"] = round(quantity * price, 2)
        enriched_positions[ticker] = market_position
        market_value += market_position["market_value"]

    total_equity = round(float(cash) + market_value, 2)
    return {
        "run_id": run_id,
        "cycle_date": cycle_date,
        "cash": round(float(cash), 2),
        "positions": enriched_positions,
        "market_value": round(market_value, 2),
        "total_equity": total_equity,
        "created_at": now,
    }


def action_for_opportunity(
    opportunity: dict[str, object],
    trades: list[dict[str, object]],
) -> str:
    ticker = opportunity["ticker"]
    for trade in trades:
        if trade["ticker"] == ticker:
            return trade["side"]
    return "hold_or_skip"


def reason_for_opportunity(
    opportunity: dict[str, object],
    trades: list[dict[str, object]],
) -> str:
    ticker = opportunity["ticker"]
    for trade in trades:
        if trade["ticker"] == ticker:
            return trade["reason"]
    return str(opportunity["reason"])


def _prediction_confidence_score(prediction: dict[str, object]) -> float:
    confidence = prediction["prediction"].get("confidence", {})
    if isinstance(confidence, dict) and confidence.get("score") is not None:
        return float(confidence["score"])
    return 0.0


def _latest_prediction_price_lookup(
    tickers: list[str],
    build_prediction_bundle: PredictionBundleBuilder,
) -> Callable[[str], Awaitable[float | None]]:
    async def lookup(ticker: str) -> float | None:
        if ticker not in tickers:
            return None
        bundle = await build_prediction_bundle(ticker)
        prediction = bundle.get("1d") or bundle.get("1w")
        if not prediction:
            return None
        return float(prediction["latest_price"])

    return lookup
