#!/usr/bin/env python3
"""Extract an actual recorded simulation frame on CPU; no policy, rendering or job submission."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0, help='Zero-based frame index')
    parser.add_argument('--output', type=Path, required=True, help='New .png file')
    args = parser.parse_args()
    if args.frame < 0 or args.output.suffix.lower() != '.png':
        parser.error('Use a nonnegative frame index and a .png output')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Output exists; choose another filename')
    import imageio.v2 as imageio
    with imageio.get_reader(args.video) as reader:
        frame = reader.get_data(args.frame)
        fps = reader.get_meta_data().get('fps')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output, frame)
    hasher = hashlib.sha256()
    with args.video.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
    metadata = {'source_video': str(args.video.resolve()), 'source_sha256': hasher.hexdigest(),
                'frame_index': args.frame, 'fps': fps, 'shape': list(frame.shape),
                'note': 'Actual recorded observation, not a generated scene. A frame alone does not prove task success.'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
