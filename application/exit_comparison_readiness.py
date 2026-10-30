"""Evidence-bound admission audit for a four-account exit experiment.

This is not a ledger or a counterfactual backtest. A saved single-account tape
cannot authorize different accounts. In particular, no hypothetical account
return is inferred by removing a recorded sell or adding its costs back.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import datetime as dt
import hashlib

from application.erem_research import compare_recorded_erem
from application.replay_contracts import digest
from application.simulation_performance_audit import audit_simulation_report
from domain.strategies.erem_soft_reduction import SoftReductionResearchPolicy


SCHEMA = "exit-four-account-readiness/v1"


def inspect_execution_source(engine_text, prices_text):
    """Report the current inspected implementation, not its historical origin.

    Unknown source shapes stay unknown. Recognizing these constants is not a
    source-authenticity, broker-fee or execution-capability certification.
    """
    engine, prices = ast.parse(engine_text), ast.parse(prices_text)
    constants = {}
    for tree in (engine, prices):
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    functions = [node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "simulate"]
    parameters = ([node.arg for node in functions[0].args.args + functions[0].args.kwonlyargs]
                  if len(functions) == 1 else [])
    # Inspect the exact residual expression, not arbitrary occurrences of 3000.
    floors = []
    if len(functions) == 1:
        for node in ast.walk(functions[0]):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            expression = ast.unparse(node.test)
            if expression.startswith("(pos['shares'] - qty) * q['open'] < "):
                value = node.test.comparators[0]
                if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                    floors.append(value.value)
    return dict(current_source_sha256=dict(
            engine=hashlib.sha256(engine_text.encode()).hexdigest(),
            execution_prices=hashlib.sha256(prices_text.encode()).hexdigest()),
        historical_source_binding="NOT_ESTABLISHED_BY_CURRENT_SOURCE_INSPECTION",
        buy_fee_rate=constants.get("BUY_FEE"), sell_combined_fee_tax_rate=constants.get("SELL_FEE"),
        residual_floor_twd=floors[0] if len(floors) == 1 else None,
        simulate_parameters=parameters,
        execution_policy=constants.get("EXECUTION_POLICY"),
        cost_contract_status="FIXED_RATES_ONLY_NOT_COMPLETE_EXECUTION_COST_EVIDENCE")


def audit_exit_comparison(csv_text, run, manifest, *, engine_text, prices_text):
    """Bind the export to its single-account source and assess permitted use.

    This format deliberately has no 'accept a ready result' shortcut. Completed
    account comparisons need actual new Bank research/execution receipts; caller
    booleans, empty action lists and saved original account inputs cannot pass.
    """
    performance = audit_simulation_report(csv_text, bank_run=run)
    diagnostic = compare_recorded_erem(run)
    if (not isinstance(manifest, dict) or manifest.get("schema") != "bank-date-current/v1"
            or run.get("evidencePolicy") != "ANALYSIS_TIME_DATA_ON_HISTORICAL_BARS/v1"):
        raise ValueError("EXIT_COMPARISON_SUPPORTED_CURRENT_DATA_ARCHIVE_REQUIRED")
    if (not isinstance(manifest, dict) or manifest.get("replayId") != run.get("jobId")
            or manifest.get("versions") != run.get("versions")
            or manifest.get("evidencePolicy") != run.get("evidencePolicy")
            or manifest.get("executionPolicy") != run.get("executionPolicy")
            or manifest.get("referenceDate") != run.get("referenceDate")
            or manifest.get("symbols") != run.get("universe")
            or manifest.get("dates") != [row["date"] for row in run["daily"]]):
        raise ValueError("EXIT_COMPARISON_MANIFEST_RUN_BINDING_MISMATCH")
    initial = run.get("initialAccount", {})
    if (initial.get("cash") != 1_000_000 or initial.get("nav") != 1_000_000
            or run.get("initialCapital") != 1_000_000
            or float(performance["account"]["initial_twd"]) != 1_000_000
            or initial.get("holdings") != []):
        raise ValueError("EXIT_COMPARISON_EXPLICIT_MILLION_EMPTY_ACCOUNT_REQUIRED")
    symbols = manifest.get("symbols")
    if (not isinstance(symbols, list) or not symbols
            or any(not isinstance(symbol, str) for symbol in symbols)
            or symbols != sorted(set(symbols))):
        raise ValueError("EXIT_COMPARISON_FIXED_UNIVERSE_REQUIRED")
    policy = SoftReductionResearchPolicy().record()
    declared = dt.datetime.fromisoformat(policy["research_declared_at"].replace("Z", "+00:00"))
    future_observations = sum(dt.datetime.fromisoformat(row["date"] + "T13:30:00+08:00") > declared
                              for row in run["daily"])
    inspected = inspect_execution_source(engine_text, prices_text)
    raw_scope = dict(market_hash=manifest.get("marketHash"), universe=symbols,
        evidence_policy=manifest["evidencePolicy"], reference_date=manifest["referenceDate"],
        source_versions=manifest["versions"], signal_start=manifest.get("firstSignal"),
        execution_dates=manifest["dates"])
    plan = dict(schema="exit-four-account-contract/v1", source_job_id=run["jobId"],
        scope=raw_scope, r1_policy=policy, initial_account=dict(cash=1_000_000,
            equity=1_000_000, holdings=[], currency="TWD", quantity_unit="SHARES"),
        entry_policy="HOLD_SOURCE_POLICY_FIXED_RECOMPUTE_ACCOUNT_BOUND_AUTHORIZATION",
        account_state_policy="INDEPENDENT_NATIVE_BANK_LEDGER_AND_NATIVE_PAE_PER_ARM",
        execution_policy="COMMON_COST_AND_CORPORATE_ACTION_CONTRACT_REQUIRED",
        residual_policy=dict(retain_floor_twd=3000, cancel="DISABLE_ONLY_RESIDUAL_FULL_LIQUIDATION",
            rounding="CEILING_NATIVE_REDUCTION_MIN_ONE_SHARE", hard_exits="UNCHANGED"))
    contract_id = digest(plan)
    arms = []
    for name, erem, residual in (("A", "NATIVE", True), ("B", "R1", True),
                                 ("C", "NATIVE", False), ("D", "R1", False)):
        arm = dict(arm=name, erem_policy=erem, retain_residual_liquidation=residual,
            contract_id=contract_id, initial_account=deepcopy(plan["initial_account"]))
        account_id = digest(arm)
        arms.append(dict(arm, account_id=account_id, pae_namespace="exit-research:" + account_id,
            status="NOT_RUN", new_account_ledger_created=False, result=None,
            net_return=None, max_drawdown=None, turnover=None, cost=None,
            corporate_action_accounting="NOT_VALIDATED"))
    gaps = [
        dict(code="ACCOUNT_BOUND_INPUTS_REQUIRE_RECOMPUTATION", owner="Crawler + BankEmulator",
             evidence="Saved EREM, nativeSizing.accountId, accountSnapshot and PAE belong to the original account.",
             required="Regenerate each arm's native inputs after every action using its own verified state."),
        dict(code="SHARED_RAW_EVIDENCE_NOT_FROZEN_FOR_FOUR_ARMS", owner="Crawler",
             evidence="The manifest binds marketHash; current-data financial/restriction collection uses its analysis clock.",
             required="Freeze common raw financial/restriction/universe evidence; do not substitute today's sources."),
        dict(code="CORPORATE_ACTION_LEDGER_NOT_VALIDATED", owner="BankEmulator + Crawler",
             evidence="Saved simulation limitations exclude dividends and share actions; final price-event evidence is not an entitlement ledger.",
             required="Verify market/category coverage, eligibility, per-share amounts/ratios, ex/record/payment dates and entitlement accounting."),
        dict(code="FULL_EXECUTION_COST_CONTRACT_MISSING", owner="BankEmulator",
             evidence="The saved result declares zero slippage; the current engine uses fixed percentage rates.",
             required="Freeze commission/minimum/rounding/tax and justified slippage/fill assumptions, with sensitivity and source evidence."),
        dict(code="FOUR_ARM_BANK_EXECUTION_NOT_RUN", owner="BankEmulator",
             evidence="Only one saved account is supplied; current native/R1/residual controls have no verified four-arm delivery receipts.",
             required="Use native Bank execution with versioned R1 and residual-only switches; independent ledgers/PAE, no copied original results."),
        dict(code="POLICY_PROMOTION_OOS_NOT_VALIDATED", owner="Research",
             evidence=f"{future_observations} saved close dates are after the declared R1 time; date alone would not certify frozen pre-decision evidence.",
             required="Collect and compare post-freeze evidence; report costs, drawdowns, turnover, exposure, fills and uncertainty.")]
    differences = [row for row in diagnostic["observed_sell_diagnostics"]
                   if row["actual_reason"] == "EREM_REDUCE"
                   and row["native_decision"] != row["research_decision"]]
    result = dict(schema=SCHEMA, status="BLOCKED_NOT_A_FOUR_ACCOUNT_BACKTEST",
        promotion_decision="DO_NOT_SWITCH_FORMAL_POLICY_ON_CURRENT_EVIDENCE",
        approved_for_live=False, email=False, order_authority=False,
        contract_id=contract_id, contract=plan, arms=arms,
        comparison_pairs=dict(exit_only=["B-A", "D-C"], residual_only=["C-A", "D-B"],
            interaction="(D-C)-(B-A)"),
        source_performance_audit=performance, recorded_account_diagnostic=diagnostic,
        first_changed_recorded_sell=deepcopy(differences[0]) if differences else None,
        observed_current_bank=inspected, source_execution_assumptions=deepcopy(run.get("executionAssumptions")),
        source_limitations=deepcopy(run.get("limitations")),
        post_r1_declaration_close_dates=future_observations,
        validated_oos_account_sessions=0, gaps=gaps,
        metric_policy="NOT_RUN_IS_NULL_NOT_ZERO_OR_CASH_ONLY_RESULT")
    result["audit_id"] = digest(result)
    # A consumer may annotate the audit without mutating saved source objects.
    return deepcopy(result)
