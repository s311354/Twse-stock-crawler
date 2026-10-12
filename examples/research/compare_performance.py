"""Full-field comparison of benchmark archives, with explicit code-ID lineage."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from application.replay_contracts import digest
from application.research_audit import compare_results
from infrastructure.research_archive import write_once


def read_day(root, measurement):
    folder = Path(measurement['archive'])
    receipt = json.loads((folder/'complete.json').read_text())
    if receipt['schema'] == 'daily-research/v1':
        return (json.loads((folder/'result.json').read_text()),
                json.loads((folder/'coverage.json').read_text()))
    def read(name):
        sha = receipt['objects'][name]
        value = json.loads((root/'job'/'objects'/sha).read_text())
        if digest(value) != sha: raise ValueError('CORRUPT_BENCHMARK_OBJECT')
        return value
    return read('result'), read('coverage')


def compare(left, right):
    a = json.loads((left/'benchmark.json').read_text())
    b = json.loads((right/'benchmark.json').read_text())
    for key in ('evidence_id', 'symbols'):
        if a[key] != b[key]: raise ValueError('BENCHMARK_INPUTS_DIFFER')
    for key in ('model', 'parameters'):
        if a['versions'][key] != b['versions'][key]: raise ValueError('BENCHMARK_MODEL_OR_PARAMETERS_DIFFER')
    days_a = {d['day']: d for d in a['measurements'] if 'day' in d}
    days_b = {d['day']: d for d in b['measurements'] if 'day' in d}
    if days_a.keys() != days_b.keys(): raise ValueError('BENCHMARK_DATES_DIFFER')
    checks, previous_map = [], {}
    for date in sorted(days_a):
        expected, expected_coverage = read_day(left, days_a[date])
        actual, actual_coverage = read_day(right, days_b[date])
        all_fields = compare_results(expected, actual)
        normalized = deepcopy(expected)
        changes = []
        if a['versions']['code'] != b['versions']['code']:
            changes.append(dict(path='versions/code', before=normalized['versions']['code'], after=b['versions']['code'],
                                reason='Measured implementation change, not a model parameter change'))
            normalized['versions']['code'] = b['versions']['code']
            previous = normalized['pae_state']['previous_result_sha256']
            if previous is not None:
                replacement = previous_map[previous]
                changes.append(dict(path='pae_state/previous_result_sha256', before=previous, after=replacement,
                                    reason='Hash of the previously fully compared result; exact lineage mapping'))
                normalized['pae_state']['previous_result_sha256'] = replacement
        business = compare_results(normalized, actual)
        coverage = compare_results(expected_coverage, actual_coverage)
        previous_map[digest(expected)] = digest(actual)
        checks.append(dict(date=date, unmodified_full_result=all_fields, metadata_derivations=changes,
            full_result_after_exact_metadata_derivation=business, full_coverage=coverage,
            coverage_status=actual_coverage['status'],
            decisions=[dict(symbol=d['symbol'], raw=d['raw'], eligible=d['research_entry_eligible'],
                            hard_gates=d['hard_gates']) for d in actual['decisions']]))
    same_version = a['versions'] == b['versions']
    return dict(schema='replay-benchmark-comparison/v1', left=str(left), right=str(right),
        same_version=same_version, days=checks,
        raw_results_equal=all(d['unmodified_full_result']['equal'] for d in checks),
        business_results_equal=all(d['full_result_after_exact_metadata_derivation']['equal'] for d in checks),
        full_coverage_equal=all(d['full_coverage']['equal'] for d in checks),
        note='All fields compared. Coverage enrichment differences remain reported, not silently accepted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path); parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    write_once(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'days'}, indent=2))
    raise SystemExit(0 if result['business_results_equal'] and (result['full_coverage_equal'] or not result['same_version']) else 1)
