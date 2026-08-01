"""Recompute per-run metrics from scratch and emit two LaTeX tables:
  A) full-run metrics, B) tracking-window (settle..divergence) metrics.
Uses the trajectory .txt + per-dataset GT + manual marks. RPE via evo.
"""
import os, sys, importlib, numpy as np
sys.path.insert(0, 'src/mTIO')
from evo.core import metrics as evom, sync
from evo.core.trajectory import PoseTrajectory3D
from mTIO.evaluation import align_and_compute_metrics

CFG = {'rovtio':'config_rovtio','sthereo':'config_sthereo',
       'firestereo':'config_firestereo','euroc':'config_euroc'}

# (dataset_label, FE, CLAHE, run_id, settle, divergence, failed)
RUNS = [
 ('ROVTIO','ORB','off','rovtio_alt1_ORB_CLAHEOFF_CLAHE-NORM',20,55,False),
 ('ROVTIO','ORB','on', 'rovtio_alt1_ORB_CLAHEON_NORM-CLAHE',16,55,False),
 ('ROVTIO','KLT','off','rovtio_alt1_KLT_CLAHEOFF_CLAHE-NORM',None,20,False),
 ('ROVTIO','KLT','on', 'rovtio_alt1_KLT_CLAHEON_NORM-CLAHE',20,50,False),
 ('STheReo','ORB','off','sthereo_valley_ORB_CLAHEOFF_NORM-CLAHE',None,120,False),
 ('STheReo','ORB','on', 'sthereo_valley_ORB_CLAHEON_NORM-CLAHE',None,95,False),
 ('STheReo','KLT','off','sthereo_valley_KLT_CLAHEOFF_NORM-CLAHE',None,270,False),
 ('STheReo','KLT','on', 'sthereo_valley_KLT_CLAHEON_NORM-CLAHE',None,None,False),
 ('FIReStereo','ORB','on','firestereo_frick1_ORB_CLAHEON_NORM-CLAHE',None,None,True),
 ('FIReStereo','KLT','on','firestereo_frick1_KLT_CLAHEON_NORM-CLAHE',None,None,True),
 ('EuRoC','ORB','off','euroc_mh03_ORB_CLAHEOFF',None,None,False),
 ('EuRoC','KLT','off','euroc_mh03_KLT_CLAHEOFF',20,None,False),
]
DS_OF = {'ROVTIO':'rovtio','STheReo':'sthereo','FIReStereo':'firestereo','EuRoC':'euroc'}
_gtcache={}
def gt_of(ds):
    if ds not in _gtcache:
        c=importlib.import_module('mTIO.'+CFG[DS_OF[ds]]); _gtcache[ds]=c.load_ground_truth()
    return _gtcache[ds]
def wxyz(q): q=np.asarray(q,float); return q[:,[3,0,1,2]]
def evo_rpe(t,xyz,quat,gt,lo=None,hi=None,gt_wxyz=False):
    m=np.ones(len(t),bool)
    if lo is not None: m&= t>=lo
    if hi is not None: m&= t<=hi
    if m.sum()<10: return None
    est=PoseTrajectory3D(positions_xyz=xyz[m],orientations_quat_wxyz=wxyz(quat[m]),timestamps=t[m])
    # EuRoC GT quaternions are already wxyz; all others are xyzw.
    gq = gt['q'] if gt_wxyz else wxyz(gt['q'])
    ref=PoseTrajectory3D(positions_xyz=gt['p'],orientations_quat_wxyz=gq,timestamps=gt['t'])
    try:
        r,e=sync.associate_trajectories(ref,est,max_diff=0.05); e.align(r,correct_scale=False)
        rp=evom.RPE(evom.PoseRelation.translation_part,delta=1.0,delta_unit=evom.Unit.meters,all_pairs=False)
        rp.process_data((r,e)); return float(rp.get_statistic(evom.StatisticsType.rmse))
    except Exception: return None

rowsA=[]; rowsB=[]
for ds,fe,cl,rid,st,dv,failed in RUNS:
    p=f'results/{rid}.txt'
    if not os.path.exists(p): print('MISSING',rid); continue
    d=np.loadtxt(p); t,xyz,quat=d[:,0],d[:,1:4],d[:,4:8]; gt=gt_of(ds)
    if failed:
        rowsA.append((ds,fe,cl,None,None,None,'fail')); continue
    m=align_and_compute_metrics(np.column_stack([t,xyz]),gt['t'],gt['p'],
                                settle_time_s=st,divergence_time_s=dv)
    ew = (ds == 'EuRoC')
    rpe_full=evo_rpe(t,xyz,quat,gt,gt_wxyz=ew)
    surv = f"{m['survival_s']:.0f}" + ('' if m['diverged'] else '*')
    rowsA.append((ds,fe,cl,m['ate_rmse_m'],rpe_full,surv,'ok'))
    # tracked window
    t0=t[0]; lo=t0+(st or 0.0); hi=t0+(dv if dv is not None else (t[-1]-t0))
    rpe_tr=evo_rpe(t,xyz,quat,gt,lo,hi,gt_wxyz=ew)
    iv=f"[{(st or 0):.0f},{dv if dv is not None else (t[-1]-t0):.0f}]"
    rowsB.append((ds,fe,cl,iv,m['pre_divergence_ate_rmse_m'],rpe_tr))

def fnum(x,fmt='{:.1f}'):
    return '--' if x is None else fmt.format(x)
print('=== TABLE A (full run) ===')
for r in rowsA:
    ds,fe,cl,ate,rpe,surv,st=r
    if st=='fail': print(f"{ds} & {fe} & {cl} & \\multicolumn{{3}}{{c}}{{\\emph{{failed to track}}}} \\\\")
    else: print(f"{ds} & {fe} & {cl} & {fnum(ate)} & {fnum(rpe,'{:.2f}')} & {surv} \\\\")
print('=== TABLE B (tracked window) ===')
for ds,fe,cl,iv,ate,rpe in rowsB:
    print(f"{ds} & {fe} & {cl} & {iv} & {fnum(ate,'{:.2f}')} & {fnum(rpe,'{:.2f}')} \\\\")