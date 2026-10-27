"""Reproduce the final incoming case from the 20:24 real recording."""
import hashlib
import json
from pathlib import Path


def main():
  source = Path('deploy/recordings/real_20261026_202459')
  target = Path('deploy/benchmarks/planner/recorded_ball_cases_20261026.json')
  summary = json.loads((source / 'summary.json').read_text())
  packet_file = source / 'mocap_packets.jsonl'
  packets = [json.loads(line) for line in packet_file.read_text().splitlines()]
  command = next(e for e in reversed(summary['events']) if e['event'] == 'motion_selected')
  balls = [(i, r) for i, r in enumerate(packets) if r['packet']['kind'] == 'ball']
  # Start at the final launch turning point before the recorded selection.
  candidates = [(j, r) for j, (_, r) in enumerate(balls)
                if command['time'] - 1.5 <= r['arrival_time'] <= command['time']]
  start = max(candidates, key=lambda item: item[1]['packet']['position'][0])[0]
  end = next(j for j in range(start + 1, len(balls))
             if balls[j][1]['packet']['position'][0] <= -0.25)
  selected = balls[start:end + 1]
  first = selected[0][1]
  stamp = first['packet']['stamp']
  # Bridge wall/monotonic epoch from the same first packet, as existing cases do.
  epoch = first['arrival_time'] - (first['wall_time'] - stamp)
  samples = [[r['packet']['stamp'] - stamp, r['arrival_time'] - epoch,
              *r['packet']['position']] for _, r in selected]
  assert all(b[0] >= a[0] and b[1] >= a[1] for a, b in zip(samples, samples[1:]))
  case = dict(id='real_20261026_202459_1', run=source.name,
    source_packet_sha256=hashlib.sha256(packet_file.read_bytes()).hexdigest(),
    source_packet_indices=[selected[0][0], selected[-1][0]],
    source_packet_index_convention='zero-based JSONL lines, inclusive endpoints',
    sample_columns=['header_relative_s', 'arrival_relative_s', 'x', 'y', 'z'],
    recorded_trigger_relative_source_s=command['time'] - epoch,
    recorded_command=command,
    cutoff_relative_trigger_s=selected[-1][1]['arrival_time'] - command['time'],
    cutoff_reason='First incoming crossing of X=-0.25m; later robot-near deflection and outgoing positions excluded.',
    coordinate_normalization=dict(source_frame='world', target_frame='hope_table',
      scale=1., quaternion_wxyz=[1., 0., 0., 0.], translation=[0., 0., 0.],
      evidence='Both real-run summaries configure identity mocap transforms. The 20:24 home anchor is [-0.5955,-0.7270,0.0488], consistent with the table frame; incoming bounce occurs near [0.916,-0.806,-0.019]. Frame label normalized; measured XYZ retained. This is a recording consistency check, not a new calibration.'),
    source_mocap_config=summary['mocap_runtime_config'], samples=samples)
  data = json.loads(target.read_text())
  data['cases'] = [c for c in data['cases'] if c['id'] != case['id']] + [case]
  data['source_recordings'] = ['deploy/recordings/real_20261026_201506', str(source)]
  data['extraction_note'] = 'Final triggered incoming trajectory from each 1026 run. The 20:24 world label is normalized using identity coordinates; case metadata records the consistency evidence. All cases stop at the incoming X=-0.25m cutoff.'
  target.write_text(json.dumps(data, indent=2) + '\n')
  print(json.dumps(dict(case=case['id'], samples=len(samples), indices=case['source_packet_indices'])))


if __name__ == '__main__':
  main()
