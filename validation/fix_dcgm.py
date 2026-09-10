from pathlib import Path
import ast
import re

root = Path('NVSentinel/health-monitors/gpu-health-monitor/gpu_health_monitor')
p = root / 'dcgm_watcher/types.py'
p.write_text(p.read_text().replace('entity_failures: dict[int, ErrorDetails]', 'entity_failures: dict[int, list[ErrorDetails]]'))
p = root / 'dcgm_watcher/dcgm.py'
s = p.read_text()
a = s.index('            suppressed_gpu_ids = [')
b = s.index('\n    def _is_nvlink_down_false_positive', a)
s = s[:a] + '''            had_failures = bool(details.entity_failures)
            for gpu_id, failures in list(details.entity_failures.items()):
                remaining = [
                    failure
                    for failure in failures
                    if not self._is_suppressed_error_code(watch_name, gpu_id, failure.code)
                ]
                if remaining:
                    details.entity_failures[gpu_id] = remaining
                else:
                    del details.entity_failures[gpu_id]

            # A watch with no remaining failures is healthy again.
            if had_failures and not details.entity_failures:
                details.status = types.HealthStatus.PASS
''' + s[b:]
a = s.index('                health_status[watch_name].status = types.HealthStatus(int(incident.health))')
b = s.index('\n            self._reset_absent_incident_streaks', a)
s = s[:a] + '''                health_status[watch_name].status = types.HealthStatus(
                    max(health_status[watch_name].status.value, int(incident.health))
                )

                # Keep messages with their code: each code determines its own remediation.
                accumulator_key = (watch_name, gpu_id, error_code)
                gpu_failures_accumulator.setdefault(accumulator_key, []).append(error_msg)

            for (watch_name, gpu_id, error_code), messages in sorted(gpu_failures_accumulator.items()):
                health_status[watch_name].entity_failures.setdefault(gpu_id, []).append(
                    types.ErrorDetails(message="; ".join(messages), code=error_code)
                )
''' + s[b:]
s = s.replace('# Temporary dict to accumulate multiple failures per GPU\n            gpu_failures_accumulator = {}', '# Group repeated incidents by watch, GPU and error code.\n            gpu_failures_accumulator: dict[tuple[str, int, str], list[str]] = {}')

def wrap_calls(text, select):
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for value in select(ast.parse(text)):
        a = offsets[value.lineno - 1] + value.col_offset
        b = offsets[value.end_lineno - 1] + value.end_col_offset
        edits.append((a, b, '[' + text[a:b] + ']'))
    for a, b, replacement in sorted(edits, reverse=True):
        text = text[:a] + replacement + text[b:]
    return text

s = wrap_calls(s, lambda tree: [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'ErrorDetails' and 'entity_failures' in ast.unparse(n.targets[0])])
p.write_text(s)
p = root / 'platform_connector/platform_connector.py'
s = p.read_text()
a = s.index('                        failure_details = details.entity_failures.get(gpu_id)')
b = s.index('\n                    else:', a)
chunk = s[a:b].replace('                        failure_details = details.entity_failures.get(gpu_id)\n', '', 1)
chunk = chunk.replace('entry = self.entity_cache.get(key)', 'entry = pending_cache_updates.get(key, self.entity_cache.get(key))', 1)
chunk = '                        for failure_details in details.entity_failures[gpu_id]:\n' + ''.join(('    ' + line if line.strip() else line) for line in chunk.splitlines(keepends=True))
p.write_text(s[:a] + chunk + s[b:])
for path in ['tests/test_dcgm_watcher/test_dcgm.py', 'tests/test_platform_connector/test_platform_connector.py']:
    p = root / path
    s = wrap_calls(p.read_text(), lambda tree: [v for n in ast.walk(tree) if isinstance(n, ast.Dict) for v in n.values if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == 'ErrorDetails'])
    s = re.sub(r'(\.entity_failures\[[^\]\n]+\])\.(code|message)', r'\1[0].\2', s)
    s = s.replace('failure = second["DCGM_HEALTH_WATCH_NVLINK"].entity_failures[3]', 'failure = second["DCGM_HEALTH_WATCH_NVLINK"].entity_failures[3][0]')
    p.write_text(s)
p = root.parent / 'README.md'
p.write_text(p.read_text() + '\n\n## Incident reporting\n\nEach GPU and watch reports every distinct error code. Repeated incidents for the same code share one event with their combined messages. Each event retains its code-specific remediation action.\n\nSuppression and debounce apply separately to each code. A healthy event clears the watch only when that GPU has no remaining reported incidents. Cache updates occur after successful delivery.\n')
