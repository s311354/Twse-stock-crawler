"""Manual isolated FR3 research. No mail, network, scheduler or account ledger."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.fr3_policy import AVAILABLE_AT, components
from infrastructure.research_archive import write_once
from infrastructure.storage.les_v7_training import read_json_bounded, MAX_CONFIG_BYTES, MAX_DATASET_BYTES


def _read(path, *, large=False):
    return read_json_bounded(path, MAX_DATASET_BYTES if large else MAX_CONFIG_BYTES)


def freeze_fr3_configs(archive, build_config_path, output):
    """Derive new configs from the old explicit settings, never from outcomes."""
    from domain.les_v7.fr3_plan import policy
    from domain.les_v7.quality_policy import r6_quality_policy
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository
    from infrastructure.les_v7_fr3_training import validate_fr3_training_configs
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output == archive.parent or output.is_relative_to(archive.parent):
        raise ValueError("FR3_NEW_OUTPUT_REQUIRED_NOT_ORIGINAL_ARCHIVE")
    verified = LesV7TrainingRepository.verify(archive.parent)
    old_bundle = _read(archive)
    if old_bundle["sha256"] != verified["bundle_sha256"]:
        raise ValueError("FR3_ORIGINAL_ARCHIVE_BINDING_MISMATCH")
    training = deepcopy(_read(archive.parent / "config.json"))
    build = deepcopy(_read(build_config_path))
    reference = policy(training["reference_policy"], available_at=AVAILABLE_AT)
    training.update(reference_policy=reference, quality_policy=r6_quality_policy())
    # The parent may have added execution-head coordinates after its original
    # raw source build. Rebuild those coordinates from raw bars for the new
    # plan; do not reuse or relabel the parent's prepared rows.
    for key in ("feature_names", "feature_bounds", "context_rules", "execution_feature_profile",
                "fill_only_feature_extension"):
        if key in training:
            build["seed"][key] = deepcopy(training[key])
        else:
            build["seed"].pop(key, None)
    build["seed"]["reference_policy"] = deepcopy(reference)
    build["path_policy"] = deepcopy(reference["cost_path_policy"])
    verified_components = validate_fr3_training_configs(build, training)
    body = dict(schema="les-v7-fr3-frozen-settings/v1", original_bundle_sha256=old_bundle["sha256"],
        build_config_id=identity(build), training_config_id=identity(training),
        components=components(), reference_components=verified_components,
        publication_at=AVAILABLE_AT, fit_partitions="UNCHANGED_FROM_PARENT_NOT_TUNED_ON_HOLDOUT",
        validation_posture="VIEWED_HOLDOUT_RETROSPECTIVE_NOT_NEW_CONFIRMATION",
        original_bundle_modified=False, ordinary_daily_modified=False,
        research_only=True, approved_for_live=False)
    record = dict(payload=body, sha256=identity(body))
    write_once(output / "build-config.json", build)
    write_once(output / "training-config.json", training)
    write_once(output / "fr3-settings.json", record)
    return record


def freeze_fr3_l2_configs(parent_training_root, output, *, available_at):
    """Freeze approved L2 rules from a verified R1 preparation's settings.

    This metadata-only entry never reads/relabels old prepared rows or fits a
    model. The later native preparation revalidates and rebuilds every frozen
    symbol from raw source. Explicit creation time makes retries deterministic;
    it is not a claim of historical deployment or source publication.
    """
    from domain.les_v7.fr3_plan import PLAN_VERSION as R1_VERSION, policy
    from domain.les_v7.fr3_l2 import PLAN_VERSION as L2_VERSION
    from infrastructure.les_v7_fr3_training import REQUEST_SCHEMA, validate_fr3_training_configs
    from infrastructure.les_v7_scope_training import DATASET_SCHEMA, inspect_elec_scope_freeze
    parent, output = Path(parent_training_root).resolve(), Path(output).resolve()
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("FR3_L2_NEW_OUTPUT_REQUIRED_NOT_PARENT_OR_ANCESTOR")
    allowed = {"build-config.json", "training-config.json", "fr3-l2-settings.json"}
    if output.exists() and (not output.is_dir() or any(p.name not in allowed for p in output.iterdir())):
        raise ValueError("FR3_L2_NEW_SETTINGS_DIRECTORY_REQUIRED")
    request = _read(parent / "fr3-training-request.json", large=True)
    body = request.get("payload", {})
    if (request.get("sha256") != identity(body) or body.get("schema") != REQUEST_SCHEMA
            or body.get("research_only") is not True or body.get("approved_for_live") is not False
            or body.get("execution_action") != "NO_ACTION"):
        raise ValueError("FR3_L2_PARENT_REQUEST_BINDING_MISMATCH")
    build, training = deepcopy(body["build_config"]), deepcopy(body["training_config"])
    parent_components = validate_fr3_training_configs(build, training)
    native_build = _read(parent / "native" / "scope-build-config.json")
    native_training = _read(parent / "native" / "scope-training-config.json")
    dataset = _read(parent / "native" / "scope-training-dataset.json", large=True)
    manifest = dataset.get("payload", {})
    scope = inspect_elec_scope_freeze(body["freeze_path"], verify_sources=False)
    if (training["reference_policy"]["version"] != R1_VERSION
            or native_build != build or native_training != training
            or body.get("components") != parent_components
            or dataset.get("sha256") != identity(manifest) or manifest.get("schema") != DATASET_SCHEMA
            or manifest.get("build_config_id") != identity(build)
            or manifest.get("training_config_id") != identity(training)
            or manifest.get("reference_config_id") != parent_components["reference_config_id"]
            or manifest.get("freeze_id") != body.get("freeze_id") or body.get("freeze_id") != scope["freeze_id"]
            or manifest.get("scope_id") != body.get("scope_id") or body.get("scope_id") != scope["scope_id"]
            or manifest.get("universe") != [[symbol, "TWSE"] for symbol in scope["symbols"]]
            or body.get("symbols") != scope["symbols"]
            or body.get("dataset_kind") != manifest.get("dataset_kind")
            or body.get("dataset_kind") != scope["dataset_kind"]
            or manifest.get("availability_policy") != body.get("availability_policy")
            or Path(manifest["freeze_path"]).resolve() != Path(body["freeze_path"]).resolve()):
        raise ValueError("FR3_L2_PARENT_R1_PREPARATION_BINDING_MISMATCH")
    created = timestamp(available_at).isoformat()
    if timestamp(created) < timestamp(training["reference_policy"]["available_at"]):
        raise ValueError("FR3_L2_POLICY_CREATION_PRECEDES_PARENT")
    old_build, old_training = deepcopy(build), deepcopy(training)
    reference = policy(training["reference_policy"], available_at=created, version=L2_VERSION)
    training["reference_policy"] = reference
    build["seed"]["reference_policy"] = deepcopy(reference)
    build["path_policy"] = deepcopy(reference["cost_path_policy"])
    verified_components = validate_fr3_training_configs(build, training)
    # Only the approved reference-rule namespace may change. In particular no
    # folds, feature selection/support, solver, minimum counts, cost or dates.
    left, right = deepcopy(training), deepcopy(old_training)
    left.pop("reference_policy"); right.pop("reference_policy")
    left_build, right_build = deepcopy(build), deepcopy(old_build)
    for item in (left_build, right_build):
        item.pop("path_policy")
        item["seed"].pop("reference_policy")
    if left != right or left_build != right_build:
        raise ValueError("FR3_L2_NON_REFERENCE_SETTINGS_CHANGED")
    settings = dict(schema="les-v7-fr3-l2-frozen-settings/v1", parent_request_id=request["sha256"],
        parent_dataset_id=dataset["sha256"], parent_training_root=str(parent),
        parent_plan_version=R1_VERSION, freeze_path=str(Path(body["freeze_path"]).resolve()),
        freeze_id=scope["freeze_id"], scope_id=scope["scope_id"], symbols=list(scope["symbols"]),
        build_config_id=identity(build), training_config_id=identity(training),
        reference_components=verified_components, publication_at=created,
        changed_fields=["training.reference_policy", "build.seed.reference_policy", "build.path_policy"],
        fit_partitions="UNCHANGED_FROM_PARENT_NOT_TUNED_ON_HOLDOUT",
        validation_posture="VIEWED_HOLDOUT_RETROSPECTIVE_NOT_NEW_CONFIRMATION",
        prepared_policy="REBUILD_ALL_FROZEN_SYMBOLS_FROM_RAW_NEVER_RELABEL_PARENT_ROWS",
        parent_prepared_rows_reused=False, parent_model_reused=False,
        source_verification="MANIFEST_IDENTITIES_ONLY_FULL_RAW_VERIFICATION_REQUIRED_BY_PREPARATION",
        original_bundle_modified=False, ordinary_daily_modified=False,
        research_only=True, approved_for_live=False)
    record = dict(payload=settings, sha256=identity(settings))
    write_once(output / "build-config.json", build)
    write_once(output / "training-config.json", training)
    write_once(output / "fr3-l2-settings.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze-configs")
    freeze.add_argument("--archive", type=Path, required=True)
    freeze.add_argument("--build-config", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze_l2 = subs.add_parser("freeze-l2-configs", help="Freeze approved L2 leader rules; never relabel old prepared rows")
    freeze_l2.add_argument("--parent-training-root", type=Path, required=True)
    freeze_l2.add_argument("--available-at", required=True, help="Explicit timezone-aware policy creation time; reuse on retry")
    freeze_l2.add_argument("--output-dir", type=Path, required=True)
    for name in ("preview", "prepare", "train"):
        sub = subs.add_parser(name)
        sub.add_argument("--freeze", type=Path, required=True)
        sub.add_argument("--build-config", type=Path, required=True)
        sub.add_argument("--training-config", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "preview": sub.add_argument("--index", type=int, default=0)
    audit = subs.add_parser("audit-sources")
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--evaluation-cutoff", required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    compare = subs.add_parser("compare")
    compare.add_argument("--archive", type=Path, required=True)
    compare.add_argument("--training-root", type=Path, required=True)
    compare.add_argument("--source-audit", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    analyze = subs.add_parser("analyze")
    analyze.add_argument("--request", type=Path, required=True)
    analyze.add_argument("--bundle", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    verify = subs.add_parser("verify")
    verify.add_argument("--training-root", type=Path, required=True)
    calendar_repair = subs.add_parser("repair-calendar", help="New source-bound calendar parser identity, no price inference")
    calendar_repair.add_argument("--calendar", type=Path, required=True)
    calendar_repair.add_argument("--output-dir", type=Path, required=True)
    expansion = subs.add_parser("expand-configs", help="Freeze outcome-blind dates from a verified official calendar")
    expansion.add_argument("--parent-training-root", type=Path, required=True)
    expansion.add_argument("--calendar", type=Path, required=True)
    expansion.add_argument("--history-start", required=True)
    expansion.add_argument("--through", required=True)
    expansion.add_argument("--output-dir", type=Path, required=True)
    prefix = subs.add_parser("prepend-sources", help="Prepend a COMPLETE official FINAL prefix; never read today's cache")
    prefix.add_argument("--freeze", type=Path, required=True)
    prefix.add_argument("--calendar", type=Path, required=True)
    prefix.add_argument("--market", type=Path, required=True)
    prefix.add_argument("--actions", type=Path, required=True)
    prefix.add_argument("--build-config", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, required=True)
    capacity = subs.add_parser("capacity", help="Necessary-count audit, not a fit or readiness approval")
    capacity.add_argument("--training-root", type=Path, required=True)
    capacity.add_argument("--output-dir", type=Path, required=True)
    fit = subs.add_parser("fit-prepared", help="Fit a new prepared expansion only after necessary counts pass")
    fit.add_argument("--training-root", type=Path, required=True)
    fit.add_argument("--audit-output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-configs":
            result = freeze_fr3_configs(args.archive, args.build_config, args.output_dir)
        elif args.command == "freeze-l2-configs":
            result = freeze_fr3_l2_configs(args.parent_training_root, args.output_dir, available_at=args.available_at)
        elif args.command == "repair-calendar":
            from infrastructure.les_v7_fr3_sources import retain_corrected_fr3_calendar
            path = retain_corrected_fr3_calendar(args.calendar, args.output_dir)
            result = dict(status="SOURCE_RETAINED_REQUIRES_RANGE_SELECTION", artifact=str(path))
        elif args.command == "expand-configs":
            from infrastructure.les_v7_fr3_sources import freeze_fr3_expansion_settings
            result = freeze_fr3_expansion_settings(args.parent_training_root, args.calendar,
                history_start=args.history_start, through=args.through, output_root=args.output_dir)
        elif args.command == "prepend-sources":
            from infrastructure.les_v7_fr3_sources import prepend_fr3_scope
            result = prepend_fr3_scope(args.freeze, args.calendar, args.market, args.actions,
                build_config=_read(args.build_config), output_root=args.output_dir,
                progress=lambda event: print(canonical(event), flush=True))
        elif args.command in {"capacity", "fit-prepared"}:
            from infrastructure.les_v7_fr3_capacity import audit_fr3_prepared_capacity, retain_fr3_capacity
            root = args.training_root.resolve()
            if args.command == "fit-prepared" and (root / "fr3-training-completion.json").exists():
                raise ValueError("FR3_EXPANSION_FIT_REQUIRES_NEW_UNCOMPLETED_PREPARATION_NO_OLD_REBASE")
            result = audit_fr3_prepared_capacity(root)
            folder, _ = retain_fr3_capacity(result,
                args.output_dir if args.command == "capacity" else args.audit_output_dir)
            if args.command == "capacity":
                result = dict(result, artifact=str(folder / "capacity.json"))
            if args.command == "fit-prepared" and result["count_capacity_allows_fit"]:
                from infrastructure.les_v7_fr3_training import train_fr3_scope
                request = _read(root / "fr3-training-request.json", large=True)["payload"]
                result = train_fr3_scope(request["freeze_path"], request["build_config"],
                    request["training_config"], root,
                    progress=lambda event: print(canonical(event), flush=True))
        elif args.command in {"prepare", "train", "preview"}:
            from infrastructure import les_v7_fr3_training as training
            build, config = _read(args.build_config), _read(args.training_config)
            if args.command == "preview":
                result = training.preview_fr3_shard(args.freeze, build, config, index=args.index)
                write_once(args.output_dir / "preview.json", result)
            else:
                def progress(event):
                    print(canonical(event), flush=True)
                action = training.prepare_fr3_scope if args.command == "prepare" else training.train_fr3_scope
                result = action(args.freeze, build, config, args.output_dir, progress=progress)
        elif args.command == "verify":
            from infrastructure.les_v7_fr3_training import verify_fr3_scope_training
            result = verify_fr3_scope_training(args.training_root)
        elif args.command == "audit-sources":
            from infrastructure.les_v7_fr3_comparison import audit_fr3_scope_sources, retain_fr3_source_audit
            result = audit_fr3_scope_sources(args.freeze, evaluation_cutoff=args.evaluation_cutoff)
            retain_fr3_source_audit(result, args.output_dir)
        elif args.command == "compare":
            from infrastructure.les_v7_fr3_comparison import compare_fr3_research, retain_fr3_comparison
            result = compare_fr3_research(args.archive, args.training_root,
                                          source_audit=_read(args.source_audit, large=True))
            retain_fr3_comparison(result, args.output_dir)
        else:
            from application.fr3_research import analyze_fr3_request, validate_fr3_report
            from infrastructure.les_v7_research_bundle import load_research_bundle
            from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
            if args.output_dir.resolve().is_relative_to(args.bundle.resolve().parent):
                raise ValueError("FR3_NEW_OUTPUT_REQUIRED_NOT_ORIGINAL_ARCHIVE")
            request = _read(args.request, large=True)
            request["model_bundle"], model_use = load_research_bundle(args.bundle, allow_fr3=True)
            universe = [[symbol, "TWSE"] for symbol in model_use["symbols"]]
            if "universe" in request and request["universe"] != universe:
                raise ValueError("FR3_REQUEST_UNIVERSE_DIFFERS_FROM_VERIFIED_ARCHIVE")
            request["universe"] = universe
            guard = ResearchVersionGuard()
            objects = ResearchObjects(args.output_dir / "objects")
            retained = guard.retain(objects)
            retained["request"] = objects.put_json(request)
            result = analyze_fr3_request(request)
            for report, item in zip(result["reports"], request["securities"]):
                validate_fr3_report(report, item, request["model_bundle"], universe=universe)
            result.pop("result_id")
            result.update(runtime=dict(versions=asdict(guard.versions), objects=retained), model_use=model_use)
            result["result_id"] = identity(result)
            guard.check()
            write_once(args.output_dir / "result.json", result)
            write_once(args.output_dir / "completion.json", dict(schema="les-v7-fr3-analysis-completion/v1",
                request_id=result["request_id"], result_id=result["result_id"],
                research_only=True, mail_sent=False, approved_for_live=False, execution_action="NO_ACTION"))
        # Summary only: never print a multi-megabyte source/evidence graph.
        summary = dict(command=args.command, status=result.get("status", "COMPLETE"),
            content_id=result.get("result_id", result.get("request_id", result.get("sha256", identity(result)))),
            research_only=True, approved_for_live=False, mail_sent=False)
        for key in ("plan_id", "dataset_id", "capacity_id", "artifact", "scope_file", "directory"):
            if key in result:
                summary[key] = result[key]
        print(canonical(summary), flush=True)
        return 2 if result.get("status") in {"NOT_READY", "NOT_ASSESSABLE",
            "INSUFFICIENT_NECESSARY_COUNTS_DO_NOT_FIT"} else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
