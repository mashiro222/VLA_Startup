"""Run the unchanged Arena evaluator with a node-local policy-server runtime."""
import argparse
import os
from pathlib import Path


def server_environment(env, local):
    env = dict(env)
    env['PYTHONPATH'] = str(Path(local) / 'site') + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run(spec_path, local):
    from sf_followup import evaluation
    from sf_recovery.workflow import check, compare
    spec = check(spec_path)
    popen = evaluation.subprocess.Popen
    def launch(cmd, *args, **kwargs):
        if 'sf_followup.evaluation' in cmd and 'server' in cmd:
            kwargs['env'] = server_environment(kwargs['env'], local)
        return popen(cmd, *args, **kwargs)
    evaluation.subprocess.Popen = launch
    try:
        evaluation.worker(Path(spec['evaluation']))
    finally:
        evaluation.subprocess.Popen = popen
    compare(spec_path)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec');p.add_argument('local')
    a=p.parse_args();run(a.spec,a.local)
