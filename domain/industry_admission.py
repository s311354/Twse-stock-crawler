"""Industry rollout policy, independent from technical entry and risk rules."""

from domain.industry_profiles import IndustryProfileMode, IndustryQualityProfile


def industry_entry_signal(signal: str, profile: IndustryQualityProfile | None) -> str:
    """Cap entry eligibility without clearing hard rejects or missing-data states."""
    if signal not in {"BUY", "EARLY_BUY", "BUY_SMALL", "WATCH", "ARMED", "WATCH_ONLY"}:
        return signal
    if profile is None:
        return "NO_MODEL"
    if profile.mode is IndustryProfileMode.SHADOW:
        return "WATCH_ONLY"
    if signal in {"BUY", "EARLY_BUY"} and not profile.allow_buy:
        return "WATCH_ONLY"
    if signal == "BUY_SMALL" and not profile.allow_buy_small:
        return "WATCH_ONLY"
    return signal


def industry_profile_metadata(profile: IndustryQualityProfile | None) -> dict[str, str]:
    return {
        "name": profile.name if profile else "UNSUPPORTED",
        "version": profile.version if profile else "N/A",
        "mode": profile.mode.value if profile else "UNSUPPORTED",
        "calibration_status": profile.calibration_status if profile else "UNKNOWN",
        "baseline_version": profile.baseline_version if profile else "N/A",
    }
