import os
from tqdm import tqdm
import psutil
import json
from pathlib import Path

os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
import ray

"""
Parallelization
"""
def to_iterator(obj_id):
    # Call this to display tqdm progressbar when using ray parallel processing
    # Source https://github.com/ray-project/ray/issues/5554
    while obj_id:
        done, obj_id = ray.wait(obj_id)
        yield ray.get(done[0])

def print_memory():
    """
    Prints memory stats
    """
    vmem = psutil.virtual_memory()
    print(f'vMem: total:{vmem.total>>30}GB, '\
        f'avail:{vmem.available>>30}GB, '\
        f'used:{vmem.used>>30}GB {vmem.percent}\%, '\
        f'slab:{vmem.slab>>30}GB, '\
        f'cached:{vmem.cached>>30}GB'
    )

def init_preprocessing(fn, parallel=False, verbose=True,
    dir_spill=None, tmpdir=None, slurm=False):
    """
    Init parallel processing
    Source: https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

    Args:
        fn (fn): Function that's to be parallelized
        tmpdir str: e.g., '/tmp/ray/' necessary on MIT Supercloud
        slurm bool: If true, gets num cpus from slurm environment variable
            instead of psutil
    Returns:
    """
    if parallel:
        if slurm:
            num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        else:
            num_threads = psutil.cpu_count(logical=True)
        
        if tmpdir:
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
        
        print(f'Parallelizing a function onto {num_threads} CPU threads using ray.')
        if not ray.is_initialized():
            if dir_spill is not None:
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem", 
                        "params": {
                            "directory_path": dir_spill # "/nobackup1/lutjens/tmp/spill"
                    }},)
                }
            else:
                _system_config=None

            #import pdb;pdb.set_trace()
            ray.init(
                _temp_dir=tmpdir,
                _system_config=_system_config,                
                num_cpus=num_threads, 
                ignore_reinit_error=True)

    if parallel:
        fn_r = ray.remote(fn).remote
    else:
        fn_r = fn

    fn_tasks = []
    return fn_r, fn_tasks

def get_parallel_fn(model_tasks, verbose=True):
    """
    Waits for parallel model tasks to finish and returns outputs
    
    Args:
        model_outputs (list(tuple)): Outputs of model
    """
    for x in tqdm(to_iterator(model_tasks), total=len(model_tasks), disable=(verbose==False)):
        pass
    model_outputs = ray.get(model_tasks) # [0, 1, 2, 3]
    return model_outputs