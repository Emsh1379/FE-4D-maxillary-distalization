import subprocess, os, shlex
PROD = dict(OFJ_GEOM='build4d/geom_fe.pkl', OFJ_ALIGNER='build4d/aligner_fe.pkl', OFJ_K_N='1000',
            OFJ_PRESCRIBE='tip', OFJ_TIP_BLEND='0.40', OFJ_BONE='rigid', OFJ_ALIGNER_TOOTH_CONTACT='1',
            PYTHONUNBUFFERED='1', OFJ_CUDSS_DIR='/kaggle/working/cudss', OFJ_CUDSS_CU='cu13')
def launch(P, job, kind, extra=None, gpu=None, threads=1, steps=None, out=None):
    W = f'/kaggle/working/ws/{P}'
    env = dict(os.environ); env.update(PROD)
    env['OFJ_PLAN_SCALE'] = open(f'{W}/build4d/plan_scale.txt').read().strip()
    for k in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): env[k] = str(threads)
    if gpu is None: env['OFJ_GPU'] = 'off'; env['CUDA_VISIBLE_DEVICES'] = ''; env['OFJ_LINSOLVE'] = 'pardiso'
    else: env['CUDA_VISIBLE_DEVICES'] = str(gpu); env['OFJ_LINSOLVE'] = 'auto'
    env.update(extra or {})
    out = out or f'build4d/rev/run_{job}.pkl'
    if kind == 'hg':
        cmd = ['python', 'ofj4d/fixed4d.py', '--out', out] + (['--iters', str(steps)] if steps else [])
    else:
        cmd = ['python', 'ofj4d/run4d.py', '--group', kind, '--out', out] + (['--steps', str(steps)] if steps else [])
    log = open(f'{W}/build4d/rev/logs/{job}.log', 'a')
    log.write(f'# {P} {job} {kind} {extra} gpu={gpu} threads={threads}\n'); log.flush()
    p = subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p.pid
