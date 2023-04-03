import re
import os, time
import argparse
import numpy as np
   
def write_job(output_filename, template_lines, models_dirs):
    with open(os.path.join(batch_jobs_dir, output_filename), mode="w+") as f:
        f.writelines(template_lines + ['\n'])
    
        for i, model_run_dir in enumerate(models_dirs):
            if filter_func(model_run_dir):
                print(f'Discarding {model_run_dir}')
                continue

            f.writelines([
                'source activate $ENV_NAME &&\\', '\n',
                f'sh {os.path.join(runs_parent_dir, model_run_dir, "run.sh")} &&\\', '\n',
	        	f'NUM_DONE=$((NUM_DONE+1)) &&\\', '\n',
                f'echo "done script $NUM_DONE/{i} : {model_run_dir}" &&\\', '\n',
                f'rm -r GLUE/data/logs/{model_run_dir}/*/checkpoint* &&\\', '\n',
                'echo "erased ckpts"', '\n\n',
            ])
            
        f.write('echo "done with all scripts"')

def get_template_job_lines():
    with open(os.path.join(batch_jobs_dir, 'gpu_job.template')) as f:
        return f.readlines()

if __name__ == "__main__":
    task_names = [
        "cola",
        "mnli",
        "mrpc", "qnli", "qqp",
        "rte", "sst2", "stsb",
        "wnli"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kwargs', default=["finetuned"],
	 type=str, nargs='+', help='keywords in filenames to exclude')
    parser.add_argument('-r', '--reverse', 
                    action='store_true', help='kwargs will be used instead of excluded')
    parser.add_argument('-s', '--split', type=int, default=1,
                    help='how many splits (scripts) to combine the jobs')
    args = parser.parse_args()

    batch_jobs_dir = 'batch_jobs'
    ts = str(int(time.time()))
    
    kwargs = "|".join(args.kwargs)
    
    runs_parent_dir = "GLUE/data/runs"
    
    if kwargs:
        if args.reverse:
            filter_func = lambda _dir: not re.search(kwargs, _dir)
            filter_type = 'exec'
        else:
            filter_func = lambda _dir: re.search(kwargs, _dir)
            filter_type = 'inc'
    else:
        filter_func = lambda _dir: False
        filter_type = ''


    template_lines = get_template_job_lines()
    models_dir_batches = np.array_split(os.listdir(runs_parent_dir), args.split)
    for task_id, task_name in enumerate(TASK_NAMES):
        template_lines_t =\
            template_lines.copy() +\
	        ['\nNUM_DONE=0\n'] +\
            [f"TASK_ID={task_id}\n"] +\
            ["export TASK_ID\n"]
    
        for batch, models_dirs in enumerate(models_dir_batches):
            output_filename =\
            f"run_GLUE_b{batch+1}of{args.split}__{task_id}_" +\
            f"{task_name}__{filter_type}_" +\
            f"{kwargs.replace('|','-')}__{ts}.job"
            write_job(output_filename, template_lines_t, models_dirs)

        
