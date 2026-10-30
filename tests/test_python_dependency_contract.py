"""Version markers preserve the old runtime and select tested 3.15 pins."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

ROOT = Path(__file__).resolve().parents[1]


def selected(lines, version):
    result = {}
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate({"python_version": version}):
            name = canonicalize_name(requirement.name)
            assert name not in result, f"duplicate effective pin: {name}"
            result[name] = str(requirement.specifier)
    return result


@pytest.mark.parametrize("version,new", [("3.11", False), ("3.14", False), ("3.15", True)])
def test_requirements_and_package_pins_match_by_interpreter(version, new):
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package = metadata["project"]
    declared = package["dependencies"] + [r for group in package["optional-dependencies"].values() for r in group]
    actual = selected((ROOT / "requirements.txt").read_text().splitlines(), version)
    assert actual == selected(declared, version)
    assert actual["numpy"] == ("==2.5.4" if new else "==2.4.4")
    assert actual["pandas"] == ("==3.0.6" if new else "==3.0.2")
    assert actual["matplotlib"] == ("==3.11.2" if new else "==3.10.8")
    assert actual["pytest"] == ("==9.1.1" if new else "==9.0.2")
    assert actual["requests"] == "==2.33.1"


def test_installed_cli_runtime_package_is_declared():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "infrastructure.runtime" in metadata["tool"]["setuptools"]["packages"]
    assert metadata["project"]["requires-python"] == ">=3.11"
