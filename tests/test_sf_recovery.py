import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sf_followup.continuation import compatible
from sf_recovery.stage import copy_verified
from sf_recovery.evaluate import server_environment
from sf_recovery import workflow


def test_retention_can_increase_without_changing_training():
    source=dict(recipe=dict(steps=60000,micro_batch=1,global_batch=32,keep_checkpoints=2,
                            learning_rate={'peak':2.5e-5}),assets={},environment={},files={})
    target=copy.deepcopy(source)
    target.update(gpus=4)
    target['recipe'].update(steps=18000,micro_batch=2,keep_checkpoints=15002)
    compatible(source,target)
    target['recipe']['keep_checkpoints']=1
    with pytest.raises(ValueError,match='retention'):compatible(source,target)
    target['recipe']['keep_checkpoints']=15002
    target['recipe']['learning_rate']['peak']=1e-4
    with pytest.raises(ValueError,match='learning-rate'):compatible(source,target)


@pytest.mark.parametrize('start,target',[(3000,18000),(5952,20952)])
def test_18000_is_retained_even_with_every_update_saved(start,target):
    keep=target-start+2
    # This is the worst case: all updates are allocation boundaries. The actual
    # trainer saves only every 1000 steps and on graceful exit.
    checkpoints=list(range(start,target+1))
    removed=checkpoints[:-keep]
    assert not removed and 18000 in checkpoints


def test_local_copy_is_verified_and_incomplete_file_is_not_accepted(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'model-data')
    dest=tmp_path/'local/model'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    copy_verified(source,dest,digest)
    assert dest.read_bytes()==source.read_bytes()
    dest.write_bytes(b'bad')
    copy_verified(source,dest,digest)
    assert dest.read_bytes()==source.read_bytes()
    with pytest.raises(ValueError,match='checksum'):
        copy_verified(source,tmp_path/'bad', '0'*64)
    assert not (tmp_path/'bad').exists()


def test_staging_preserves_executable_mode(tmp_path):
    source=tmp_path/'tool';source.write_bytes(b'tool');source.chmod(0o755)
    dest=tmp_path/'copy';copy_verified(source,dest)
    assert dest.stat().st_mode & 0o777 == 0o755


def test_server_keeps_original_environment_and_prepends_local_runtime():
    old={'PYTHONPATH':'/repo/src:/upstream','OMP_NUM_THREADS':'2'}
    new=server_environment(old,'/tmp/job')
    assert new['PYTHONPATH']=='/tmp/job/site:/repo/src:/upstream'
    assert new['OMP_NUM_THREADS']=='2' and old['PYTHONPATH']=='/repo/src:/upstream'


@pytest.mark.parametrize('stage,gpus,nodes',[('continue',2,2),('eval',1,1)])
def test_submission_only_contains_requested_sf_job(tmp_path,monkeypatch,stage,gpus,nodes):
    monkeypatch.setattr(workflow,'check',lambda _: {})
    recorded=[]
    monkeypatch.setattr(workflow,'submit_once',lambda directory,label,cmd: recorded.append(cmd) or '123')
    assert workflow.submit(tmp_path/'recovery.json',stage,True)=='123'
    cmd=recorded[0]
    assert f'--gres=gpu:a100:{gpus}' in cmd and f'--nodes={nodes}' in cmd
    assert '--account=gts-jli741-paid' in cmd
    assert not any(x.startswith(('--dependency=','--array=')) for x in cmd)
    assert 'baseline' not in ' '.join(cmd)


def test_failed_checkpoint_audit_blocks_preparation(tmp_path):
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError,match='restore audit'):
        workflow.prepare(tmp_path/'run',audit)
    assert not (tmp_path/'run').exists()


def test_diagnostic_wrapper_preserves_forward_gradients_and_checkpoint_selection(tmp_path,monkeypatch):
    import torch
    from torch import nn
    from sf_recovery.train import install
    import sf_recovery.train as recovery
    monkeypatch.setattr(recovery.faulthandler,'dump_traceback_later',lambda *a,**k:None)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.tensor(2.0))
        def forward(self,batch):
            loss=(self.weight*batch).square().mean()
            return loss,loss.detach(),loss.detach()*0
    class Data:
        def __init__(self,root,*a,**k):self.root=root
        def __getitem__(self,index):return index
    calls=[]
    trainer=SimpleNamespace(ArenaData=Data,build_policy=lambda cfg,path:path,build_teacher=lambda path:path,
                            batch_indices=lambda *a,**k:[1],Experiment=Model,
                            checkpoint=SimpleNamespace(restore=lambda path,*a,**k:calls.append(path) or 3000))
    source=tmp_path/'source.pt'
    (tmp_path/'resume-source.json').write_text(json.dumps({'source':str(source.resolve())}))
    install(trainer,tmp_path,tmp_path/'phase.json')
    reference=Model();traced=trainer.Experiment();batch=torch.tensor([1.,2.])
    reference(batch)[0].backward();traced(batch)[0].backward()
    torch.testing.assert_close(reference.weight.grad,traced.weight.grad,rtol=0,atol=0)
    assert json.loads((tmp_path/'phase.json').read_text())['phase']=='backward_entered'
    assert trainer.ArenaData('unused').root==tmp_path/'arena_l0'
    assert trainer.checkpoint.restore(source)==3000 and calls==[tmp_path/'resume.pt']
    with pytest.raises(ValueError,match='selected latest'):
        trainer.checkpoint.restore(tmp_path/'wrong.pt')


@pytest.mark.parametrize('corrupt',[False,True])
def test_real_cpu_audit_restores_adam_and_rejects_nonfinite_weights(tmp_path,corrupt):
    import torch
    from torch import nn
    from sf_training.checkpoint import save,rng_state
    from sf_training.lora import parameter_manifest
    from sf_training.utils import sha
    from sf_recovery.audit import audit
    base=tmp_path/'.runtime/training/pi05_base_torch/conversion.json'
    base.parent.mkdir(parents=True);base.write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'root':str(tmp_path)}))
    model=nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).square().mean().backward();opt.step()
    for state in opt.state.values():state['step'].fill_(3000)
    if corrupt:
        with torch.no_grad():model.weight[0,0]=float('nan')
    checkpoint=tmp_path/'step-00003000.pt'
    identity=dict(plan_sha256=sha(plan),arm='sf',world=2,base=sha(base))
    save(checkpoint,model,opt,3000,identity,[rng_state(),rng_state()])
    (tmp_path/'parameters.json').write_text(json.dumps(parameter_manifest(model)))
    if corrupt:
        with pytest.raises(ValueError,match='Nonfinite'):
            audit(plan,checkpoint,tmp_path/'audit.json')
    else:
        report=audit(plan,checkpoint,tmp_path/'audit.json')
        assert report['passed'] and report['production_restore_passed']
        assert report['optimizer_states_restored']==2


def test_migration_preserves_weights_and_adam_while_rebuilding_four_rank_rng(tmp_path,monkeypatch):
    import torch
    from torch import nn
    from sf_followup import continuation
    from sf_training import workflow as training_workflow
    from sf_training.checkpoint import save,rng_state
    from sf_training.utils import sha
    source=tmp_path/'source/plan.json';source.parent.mkdir()
    cfg=dict(steps=60000,micro_batch=1,global_batch=32,keep_checkpoints=2,seed=42)
    source.write_text(json.dumps(dict(recipe=cfg,gpus=2,assets={},environment={},files={})))
    calibration=tmp_path/'calibration';calibration.mkdir()
    (calibration/'calibration.json').write_text(json.dumps({'passed':True,'rows':[{}, {}, {}, {}]}))
    pinned=tmp_path/'pinned';(pinned/'sf').mkdir(parents=True)
    base=tmp_path/'.runtime/training/pi05_base_torch/conversion.json'
    base.parent.mkdir(parents=True);base.write_text('{}')
    model=nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward();opt.step()
    for state in opt.state.values():state['step'].fill_(3000)
    identity=dict(plan_sha256=sha(source),arm='sf',world=2,base=sha(base))
    source_checkpoint=pinned/'sf/step-00003000.pt'
    save(source_checkpoint,model,opt,3000,identity,[rng_state(),rng_state()])
    output=tmp_path/'continued'
    continuation.prepare(source,output,calibration,pinned,18000,3000)
    monkeypatch.setattr(continuation,'ROOT',tmp_path)
    monkeypatch.setattr(training_workflow,'verify_plan',lambda path:json.loads(path.read_text()))
    continuation.migrate(output/'plan.json','sf')
    result=torch.load(output/'sf/step-00003000.pt',weights_only=False)
    original=torch.load(source_checkpoint,weights_only=False)
    assert result['step']==3000 and result['identity']['world']==4
    assert result['identity']['plan_sha256']==sha(output/'plan.json')
    assert len(result['rng'])==4
    assert not torch.equal(result['rng'][0]['torch'],result['rng'][1]['torch'])
    for name,value in original['trainable'].items():
        torch.testing.assert_close(result['trainable'][name],value,rtol=0,atol=0)
    for ident,state in original['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(result['optimizer']['state'][ident][key],value,rtol=0,atol=0)


def test_audit_accepts_only_previously_recorded_unused_parameters(tmp_path):
    import torch
    from torch import nn
    from sf_training.checkpoint import save,rng_state
    from sf_training.lora import parameter_manifest
    from sf_training.utils import sha
    from sf_recovery.audit import audit
    base=tmp_path/'.runtime/training/pi05_base_torch/conversion.json'
    base.parent.mkdir(parents=True);base.write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'root':str(tmp_path)}))
    model=nn.Module();model.unused=nn.Parameter(torch.ones(1));model.layer=nn.Linear(2,1)
    opt=torch.optim.AdamW(model.parameters())
    model.layer(torch.ones(1,2)).sum().backward();opt.step()
    for state in opt.state.values():state['step'].fill_(3000)
    checkpoint=tmp_path/'step-00003000.pt'
    save(checkpoint,model,opt,3000,dict(plan_sha256=sha(plan),arm='sf',world=2,base=sha(base)),[rng_state(),rng_state()])
    (tmp_path/'parameters.json').write_text(json.dumps(parameter_manifest(model)))
    recorded=tmp_path/'gradients-step-1.json'
    recorded.write_text(json.dumps({'without_grad':['unused']}))
    assert audit(plan,checkpoint,tmp_path/'audit.json')['recorded_unused_parameters']==['unused']
    recorded.write_text(json.dumps({'without_grad':[]}))
    with pytest.raises(ValueError,match='Missing Adam states'):
        audit(plan,checkpoint,tmp_path/'bad-audit.json')
