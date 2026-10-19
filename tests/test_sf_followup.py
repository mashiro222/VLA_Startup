import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from arena_startup.core import save
from sf_followup.checkpoints import pin
from sf_followup.continuation import compatible
from sf_followup.control import command, submit_once
from sf_followup.evaluation import rng_seed, input_batch, validate_checkpoint, matched_units


def test_pin_survives_original_retention(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'pinned'
    save(source / 'plan.json', {'recipe': {'global_batch': 32}})
    (source / 'baseline').mkdir()
    original = source / 'baseline/step-00005000.pt'
    original.write_bytes(b'completed checkpoint')
    (source / 'baseline/step-00003000.tmp').write_bytes(b'incomplete')
    result = pin(source, output)
    assert [(x['arm'], x['step']) for x in result] == [('baseline', 5000)]
    original.unlink()
    assert (output / 'baseline/step-00005000.pt').read_bytes() == b'completed checkpoint'
    # A missing original must not hide an existing pinned checkpoint on the next scan.
    assert pin(source, output)[0]['step'] == 5000


def test_migration_preserves_experiment_budget():
    source = {'recipe': {'global_batch': 32, 'micro_batch': 1, 'steps': 60000,
                         'learning_rate': {'peak': 2.5e-5}}, 'assets': {}, 'environment': {}, 'files': {}}
    target = copy.deepcopy(source)
    target.update(gpus=4); target['recipe'].update(micro_batch=2, steps=20000)
    compatible(source, target)
    target['recipe']['learning_rate']['peak'] *= 2
    with pytest.raises(ValueError, match='learning-rate'): compatible(source, target)


def test_rng_replay_and_advancement():
    seed, following = rng_seed([0, 123])
    assert rng_seed([0, 123]) == (seed, following)
    assert following == [1, 123] and rng_seed(following)[0] != seed
    with pytest.raises(ValueError): rng_seed([0, -1])
    with pytest.raises(ValueError): rng_seed([0.0, 123])


def test_inference_matches_training_input_conventions():
    import numpy as np
    from sf_training.normalization import normalize
    class Tokenizer:
        def encode(self, text, add_bos=False):
            return ([2] if add_bos else []) + [ord(c) for c in text]
    image = np.zeros((224, 224, 3), np.uint8); image[0, 0] = [10, 20, 30]
    state = np.arange(8, dtype=np.float32)
    stats = {'state': {'q01': [-1]*8, 'q99': [10]*8}}
    obs = {'observation/image': image, 'observation/wrist_image': image,
           'observation/state': state, 'prompt': ' pick_up\nbowl '}
    batch = input_batch(obs, Tokenizer(), stats, 32)
    assert batch['images'].shape == (1, 2, 3, 224, 224)
    np.testing.assert_allclose(batch['images'][0, 0, :, 0, 0], [10/255, 20/255, 30/255])
    np.testing.assert_allclose(batch['state'][0, :8], normalize(state, stats['state']))
    assert (batch['state'][0, 8:] == 0).all()
    ids = [2] + list(map(ord, 'pick up bowl')) + [10]
    assert batch['tokens'][0, :len(ids)].tolist() == ids
    assert batch['token_mask'].sum() == len(ids)


def test_four_gpu_request_is_two_nodes_and_no_memory_constraint():
    cmd = command('test', 'slurm/sf_continue.sbatch', ['plan', 'sf', 'campaign'], gpus=2, nodes=2,
                  dependencies=['123', '456'])
    assert '--nodes=2' in cmd and '--ntasks=2' in cmd and '--gres=gpu:a100:2' in cmd
    assert '--dependency=afterany:123:456' in cmd
    assert not any(x.startswith('--constraint=') for x in cmd)
    assert '--account=gts-jli741-paid' in cmd


def test_uncertain_submission_never_retries(tmp_path):
    result = mock.Mock(returncode=1, stdout='', stderr='Socket timed out')
    with mock.patch('sf_followup.control.subprocess.run', return_value=result) as run:
        with pytest.raises(RuntimeError): submit_once(tmp_path, 'calibration', ['sbatch'])
        with pytest.raises(RuntimeError, match='Uncertain'): submit_once(tmp_path, 'calibration', ['sbatch'])
        assert run.call_count == 1


def test_successful_submission_is_idempotent(tmp_path):
    result = mock.Mock(returncode=0, stdout='12345\n', stderr='')
    with mock.patch('sf_followup.control.subprocess.run', return_value=result) as run:
        assert submit_once(tmp_path, 'calibration', ['sbatch']) == '12345'
        assert submit_once(tmp_path, 'calibration', ['sbatch']) == '12345'
        assert run.call_count == 1


@pytest.mark.parametrize('key,value', [('step', 6000), ('arm', 'sf'), ('plan_sha256', 'other'), ('base', 'other')])
def test_checkpoint_cannot_silently_change_budget_or_model(key, value):
    spec = dict(plan_sha256='plan', arm='baseline', step=5000)
    ck = dict(step=5000, identity=dict(plan_sha256='plan', arm='baseline', base='base'))
    validate_checkpoint(ck, spec, 'base')
    if key == 'step': ck[key] = value
    else: ck['identity'][key] = value
    with pytest.raises(ValueError, match='identity/step'): validate_checkpoint(ck, spec, 'base')


def test_controller_refreshes_new_evaluation_jobs_before_four_gpu_submissions(tmp_path):
    from sf_followup import control
    source = tmp_path / 'source'; source.mkdir()
    probe = tmp_path / 'probe'; save(probe / 'calibration.json', {'passed': True})
    c = dict(source_plan=str(source / 'plan.json'), pinned=str(tmp_path / 'pinned'),
             original_jobs={'baseline': 'old-base', 'sf': 'old-sf'}, initial_step=5000, target_step=20000,
             calibration=str(probe), continuation_plan=str(tmp_path / 'future/plan.json'),
             evaluations={'baseline': str(tmp_path / 'eval-base'), 'sf': str(tmp_path / 'eval-sf')})
    calls = []
    for run in c['evaluations'].values():
        save(Path(run) / 'plan.json', {'time_limit': '24:00:00'})
    def submitted(directory, label, cmd):
        calls.append((label, cmd)); return {'eval-baseline': '101', 'eval-sf': '102'}.get(label, '200')
    with mock.patch.object(control, 'check', return_value=c), \
         mock.patch.object(control, 'pin', return_value=[dict(arm='baseline', step=5000), dict(arm='sf', step=5000)]), \
         mock.patch.object(control, 'active_jobs', side_effect=[set(), {'101', '102'}]), \
         mock.patch.object(control, 'receipts', return_value={}), \
         mock.patch.object(control, 'submit_once', side_effect=submitted):
        control.tick(tmp_path / 'campaign.json')
    base = next(cmd for label, cmd in calls if label.startswith('continue-baseline'))
    sf = next(cmd for label, cmd in calls if label.startswith('continue-sf'))
    assert not any(x.startswith('--dependency=') for x in base)
    assert '--dependency=afterany:101:102' in sf
    for label, cmd in calls:
        if label.startswith('eval-'):
            assert '--time=24:00:00' in cmd and '--gres=gpu:a100:1' in cmd


@pytest.fixture
def evaluation_catalog():
    return dict(config={'evaluation': dict(init_state_selection_mode='episode_idx', init_state_offset=0)},
                units=[dict(suite=suite, level=level, task_id=task, id=f'{suite}-{level}-{task}',
                            init_count=50, seed=7, trials=20)
                       for suite in ('safety_static_obstacles', 'safety_dynamic_obstacles')
                       for level in range(3) for task in range(5)])


def test_900_episodes_cover_identical_tasks_and_initial_states(evaluation_catalog):
    from collections import Counter
    baseline = matched_units(evaluation_catalog, 30)
    sf = matched_units(evaluation_catalog, 30)
    assert baseline == sf
    assert len({(u['id'], i) for u in baseline for i in range(u['trials'])}) == 900
    by_suite = Counter()
    by_level = Counter()
    for unit in baseline:
        by_suite[unit['suite']] += unit['trials']
        by_level[(unit['suite'], unit['level'])] += unit['trials']
    assert set(by_suite.values()) == {450} and set(by_level.values()) == {150}
    assert all(u['trials'] == 20 for u in evaluation_catalog['units'])


@pytest.mark.parametrize('offset,trials', [(0, 51), (21, 30), (0, 0), (0, -1)])
def test_evaluation_rejects_repeated_or_invalid_initial_states(evaluation_catalog, offset, trials):
    evaluation_catalog['config']['evaluation']['init_state_offset'] = offset
    with pytest.raises(ValueError): matched_units(evaluation_catalog, trials)


def test_evaluation_rejects_missing_task_even_if_total_count_is_30(evaluation_catalog):
    evaluation_catalog['units'][0] = copy.deepcopy(evaluation_catalog['units'][1])
    with pytest.raises(ValueError, match='30 distinct'): matched_units(evaluation_catalog, 30)
