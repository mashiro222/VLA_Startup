#!/usr/bin/env python3
"""Extract local contact sheets from existing videos; no simulation or submission."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena_startup.core import ROOT, run_path


def main():
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    run = run_path(args.run_id)
    plan = json.loads((run / 'plan.json').read_text())
    output = ROOT / 'reports' / args.run_id / 'diagnostic_frames'
    output.mkdir(parents=True, exist_ok=True)
    for suite in plan['config']['suites']:
        units = [u for u in plan['units'] if u['suite'] == suite and u['task_id'] == 0]
        units = sorted({u['level']: u for u in reversed(units)}.values(), key=lambda u: u['level'])
        canvas = Image.new('RGB', (6*224, len(units)*250), 'white')
        draw = ImageDraw.Draw(canvas)
        found = False
        for i, unit in enumerate(units):
            path = run / 'units' / unit['id'] / 'videos/0.mp4'
            if not path.exists():
                draw.text((5, i*250+5), f"L{unit['level']}: no recorded video", fill='black')
                continue
            found = True
            with imageio.get_reader(path) as reader:
                n = reader.count_frames()
                for j, index in enumerate(np.linspace(0, n-1, 6, dtype=int)):
                    canvas.paste(Image.fromarray(reader.get_data(int(index))).resize((224,224)), (j*224,i*250+26))
                    draw.text((j*224+4,i*250+5), f"L{unit['level']} pre-step {index+1}", fill='black')
        if found:
            canvas.save(output / f'{suite}.jpg')
            print(output / f'{suite}.jpg')


if __name__ == '__main__':
    main()
