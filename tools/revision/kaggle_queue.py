"""Revision job queue v2 (resumable; adopts jobs started by v1). GPU workers run the aligner jobs, then
help with the headgear jobs."""
import sys, os, time, pickle, threading, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kaggle_launch as launch
PAT = ['P8', 'P3', 'P4', 'P5', 'P6']
IZC_ANGLE = {'P8': 25.0, 'P3': 26.3, 'P4': 19.7, 'P5': 20.4, 'P6': 24.5}   # IZC elastic, deg above the occlusal plane
LEVEL = {'OFJ_WIRE_SETUP': 'level', 'OFJ_FIXED': 'build4d/rev/fixed_level.pkl'}
GPU_JOBS = [(P, 'ctl_pdl068', 'ca_control', {'OFJ_PDL_LINEAR_E': '0.68'}, 70) for P in PAT]
CPU_JOBS = ([(P, 'hg_izc', 'hg', {'OFJ_HG_ANGLE': f'{-IZC_ANGLE[P]}'}, 140) for P in PAT]
            + [(P, 'hg_level', 'hg', LEVEL, 140) for P in PAT]
            + [(P, 'hg_cres', 'hg', {'OFJ_HG_LINE': 'cres'}, 140) for P in PAT])
lock = threading.Lock(); taken = set()
LOG = '/kaggle/working/queue.log'
def note(s): open(LOG, 'a').write(f'{time.ctime()} {s}\n')
def W(P): return f'/kaggle/working/ws/{P}'
def done(P, job, n):
    f = f'{W(P)}/build4d/rev/run_{job}.pkl'
    try: D = pickle.load(open(f, 'rb'))
    except Exception: return False
    return D.get('iter', D.get('step')) >= n
def live(P, job):
    f = f'{W(P)}/build4d/rev/logs/{job}.log'
    return os.path.exists(f) and time.time() - os.path.getmtime(f) < 240
def wait_pid(pid):
    while True:
        try:
            r = os.waitpid(pid, os.WNOHANG)
            if r[0] != 0: return
        except ChildProcessError:
            try: os.kill(pid, 0)
            except ProcessLookupError: return
        time.sleep(10)
def run_job(j, gpu):
    P, job, kind, extra, n = j
    if live(P, job):                                   # started by the previous queue: wait for it
        note(f'{P} {job} adopted'); 
        while live(P, job) and not done(P, job, n): time.sleep(30)
    for attempt in range(3):
        if done(P, job, n): break
        if extra.get('OFJ_FIXED') and not os.path.exists(f"{W(P)}/{extra['OFJ_FIXED']}"):
            env = dict(os.environ); env.update(launch.PROD); env.update(extra); env['OFJ_GPU'] = 'off'
            r = subprocess.run(['python', 'ofj4d/fixed4d.py', '--build'], cwd=W(P), env=env, capture_output=True, text=True)
            open(f'{W(P)}/build4d/rev/logs/build_{job}.log', 'w').write(r.stdout + r.stderr)
        wait_pid(launch.launch(P, job, kind, extra, gpu=gpu, threads=1))
    note(f'{P} {job} finished={done(P, job, n)}')
def worker(lists, gpu):
    for jobs in lists:
        while True:
            with lock:
                todo = [j for j in jobs if (j[0], j[1]) not in taken and not done(j[0], j[1], j[4])]
                if not todo: break
                j = todo[0]; taken.add((j[0], j[1]))
            run_job(j, gpu)
note('queue v3 start (cuDSS)')
T = [threading.Thread(target=worker, args=([GPU_JOBS, CPU_JOBS], 0)), threading.Thread(target=worker, args=([GPU_JOBS, CPU_JOBS], 1)),
     threading.Thread(target=worker, args=([CPU_JOBS], 0)), threading.Thread(target=worker, args=([CPU_JOBS], 1))]
for t in T: t.start(); time.sleep(3)
for t in T: t.join()
note('ALL DONE')
