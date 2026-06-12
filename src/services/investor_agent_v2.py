from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from uuid import uuid4

from src.services.investor_agent import (
    DAILY_DEPLOY_CASH_FRACTION,
    INITIAL_CASH,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_FRACTION,
    MIN_BUY_CONFIDENCE,
    MIN_BUY_SCORE,
    MIN_TRADE_AMOUNT,
    PredictionBundleBuilder,
    action_for_opportunity,
    allocate_cash,
    as_utc_datetime,
    build_investment_opportunity,
    build_portfolio_snapshot,
    buy_position,
    merge_position,
    reason_for_opportunity,
    sell_position,
)


RUN_DAYS = 5


async def start_agent_v2_run(
    db,
    *,
    initial_cash: float = INITIAL_CASH,
    duration_days: int = RUN_DAYS,
    now: datetime | None = None,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    existing = await db.agent_v2_runs.find_one({"status": "active"}, {"_id": 0})
    if existing:
        return {
            "status": "already_active",
            "run": existing,
        }

    run = {
        "run_id": str(uuid4()),
        "agent_version": "v2_top_up",
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
            "held_position_top_ups": True,
        },
        "created_at": now,
        "updated_at": now,
    }
    await db.agent_v2_runs.insert_one(run)
    run.pop("_id", None)
    return {
        "status": "started",
        "run": run,
    }


async def get_agent_v2_status(db) -> dict[str, object]:
    run = await db.agent_v2_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        run = await db.agent_v2_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    if not run:
        return {"status": "not_started"}

    snapshot = await db.agent_v2_portfolio_snapshots.find_one(
        {"run_id": run["run_id"]},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    return {
        "status": run["status"],
        "run": run,
        "latest_snapshot": snapshot,
    }


async def get_agent_v2_history(db, *, limit: int = 100) -> dict[str, object]:
    run = await db.agent_v2_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    if not run:
        return {"status": "not_started", "run": None, "decisions": [], "trades": [], "snapshots": []}

    run_id = run["run_id"]
    decisions = await db.agent_v2_decisions.find(
        {"run_id": run_id},
        {"_id": 0},
    ).sort("created_at", -1).limit(limit).to_list(length=limit)
    trades = await db.agent_v2_trades.find(
        {"run_id": run_id},
        {"_id": 0},
    ).sort("created_at", -1).limit(limit).to_list(length=limit)
    snapshots = await db.agent_v2_portfolio_snapshots.find(
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


async def run_daily_agent_v2_cycle(
    db,
    *,
    tickers: list[str],
    build_prediction_bundle: PredictionBundleBuilder,
    now: datetime | None = None,
    force: bool = False,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    run = await db.agent_v2_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        return {"status": "error", "message": "No active agent v2 run found."}

    if now >= as_utc_datetime(run["ends_at"]):
        return await liquidate_agent_v2_run(
            db,
            price_lookup=_latest_prediction_price_lookup(tickers, build_prediction_bundle),
            now=now,
            reason="scheduled_end",
        )

    cycle_date = now.date().isoformat()
    existing_snapshot = await db.agent_v2_portfolio_snapshots.find_one(
        {"run_id": run["run_id"], "cycle_date": cycle_date},
        {"_id": 0},
    )
    if existing_snapshot and not force:
        return {
            "status": "already_ran_today",
            "snapshot": existing_snapshot,
        }

    opportunities = []
    decisions = []
    for ticker in tickers:
        try:
            bundle = await build_prediction_bundle(ticker)
            opportunities.append(build_investment_opportunity(ticker, bundle["1d"], bundle["1w"]))
        except Exception as exc:
            opportunities.append(
                {
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
            )

    run = await db.agent_v2_runs.find_one({"run_id": run["run_id"]}, {"_id": 0})
    cash = float(run["cash"])
    positions = dict(run.get("positions", {}))
    initial_cash = float(run["initial_cash"])
    trades = []

    for opportunity in opportunities:
        ticker = opportunity["ticker"]
        position = positions.get(ticker)
        if not position or opportunity.get("latest_price") is None:
            continue
        if float(opportunity["combined_score"]) <= 0.0:
            trade = sell_position(
                run_id=run["run_id"],
                ticker=ticker,
                position=position,
                price=float(opportunity["latest_price"]),
                now=now,
                reason="model_score_not_positive",
            )
            cash += trade["cash_delta"]
            positions.pop(ticker, None)
            trades.append(trade)

    buy_candidates = select_agent_v2_buy_candidates(opportunities, positions)
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
            reason=(
                "top_up_existing_high_confidence_position"
                if ticker in positions
                else "high_confidence_positive_model_signal"
            ),
        )
        cash -= buy_amount
        positions[ticker] = merge_position(positions.get(ticker), trade)
        trades.append(trade)

    for opportunity in opportunities:
        decisions.append(
            {
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
        )

    if decisions:
        await db.agent_v2_decisions.insert_many(decisions)
    if trades:
        await db.agent_v2_trades.insert_many(trades)

    snapshot = build_portfolio_snapshot(
        run_id=run["run_id"],
        cycle_date=cycle_date,
        cash=cash,
        positions=positions,
        opportunities=opportunities,
        now=now,
    )
    await db.agent_v2_portfolio_snapshots.insert_one(snapshot)
    await db.agent_v2_runs.update_one(
        {"run_id": run["run_id"]},
        {
            "$set": {
                "cash": round(cash, 2),
                "positions": snapshot["positions"],
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


async def liquidate_agent_v2_run(
    db,
    *,
    price_lookup: Callable[[str], Awaitable[float | None]],
    now: datetime | None = None,
    reason: str = "manual_liquidation",
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    run = await db.agent_v2_runs.find_one({"status": "active"}, {"_id": 0})
    if not run:
        return {"status": "error", "message": "No active agent v2 run found."}

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
        await db.agent_v2_trades.insert_many(trades)

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
    await db.agent_v2_portfolio_snapshots.insert_one(snapshot)
    await db.agent_v2_runs.update_one(
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


def select_agent_v2_buy_candidates(
    opportunities: list[dict[str, object]],
    positions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    held = set(positions)
    held_candidates = [
        opportunity
        for opportunity in opportunities
        if opportunity["eligible"] and opportunity["ticker"] in held
    ]
    new_candidates = [
        opportunity
        for opportunity in opportunities
        if opportunity["eligible"] and opportunity["ticker"] not in held
    ]
    new_candidates.sort(key=lambda item: item["investment_score"], reverse=True)
    remaining_slots = max(0, MAX_OPEN_POSITIONS - len(held))
    candidates = [*held_candidates, *new_candidates[:remaining_slots]]
    candidates.sort(key=lambda item: item["investment_score"], reverse=True)
    return candidates


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
