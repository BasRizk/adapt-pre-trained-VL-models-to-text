import re
import os, time
import argparse
import numpy as np

def write_job(output_filename, base_template_lines, runs):

    with open(os.path.join(batch_jobs_dir, output_filename), mode="w+") as f:
        f.writelines(base_template_lines + ['\n'])
        for i, (_dir, task_id, task_name) in enumerate(runs):
            lines = [
                f'\n\necho "evaluating {_dir} on {task_id}:{task_name}"\n',
                f"TASK_ID={task_id}\n"
                "export TASK_ID\n", 
                'source activate $ENV_NAME &&\\', '\n',
                f'sh {os.path.join(runs_parent_dir, _dir, "run.sh")} &&\\', '\n',
	        	f'NUM_DONE=$((NUM_DONE+1)) &&\\', '\n',
                f'echo "done script $NUM_DONE/{i+1} : {_dir}"',
            ]
            # if next directory is different .. otherwise keep ckpts
            if args.delete_ckpt_reg or (len(runs) > i+1 and runs[i+1][0] != _dir):
                lines += [
                    '&&\\', '\n',
                    f'rm -r GLUE/data/logs/{_dir}/*/checkpoint* &&\\', '\n',
                    'echo "erased ckpts"', '\n\n',
                ]
            f.writelines(lines)
            
        f.write(f'echo "done trying to run all {len(runs)} scripts"')

def get_completed_predictions():
    finished_preds = set()
    for root, dirs, files in os.walk('GLUE/data/logs', topdown=False):
        for name in files:
            if name.startswith('predict_results'):
                task = re.match(r'predict_results_(.+).txt', name).groups()[0]
                model = root.split('/')[3]
                # for bert-base-uncased
                model = model.replace('-uncased', '')
                finished_preds.add((model, task))
    return list(finished_preds)
    
def get_template_job_lines():
    with open(os.path.join(batch_jobs_dir, 'gpu_job.template')) as f:
        return f.readlines()

if __name__ == "__main__":
    _TASK_NAMES = {
        0:"cola",
        1:"mnli",
        2:"mrpc",
        3:"qnli",
        4:"qqp",
        5:"rte",
        6:"sst2",
        7:"stsb",
        8:"wnli"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_ids', default=_TASK_NAMES.keys(),
                        type=str, nargs='+',
                        help=f'task names to include {str(_TASK_NAMES)}')
    parser.add_argument('-k', '--kwargs', default=[],
                        type=str, nargs='+', 
                        help='keywords in filenames to exclude')
    parser.add_argument('-r', '--reverse', 
                    action='store_true', help='kwargs will be used instead of excluded')
    parser.add_argument('-d', '--delete_ckpt_reg', 
                    action='store_true', help='delete ckpt regularly or after every few runs')
    parser.add_argument('-ignore', '--ignore_completed', 
                    action='store_true', help='discard models with tasks where predicted_results_TASK.txt exist')
    parser.add_argument('-s', '--split', type=int, default=1,   
                    help='how many splits (scripts) to combine the jobs')
    parser.add_argument('-b_dir', type=str, default='batch_jobs')
    parser.add_argument('-r_dir', type=str, default='GLUE/data/runs')
    args = parser.parse_args()

    logs_dir = 'GLUE/data/logs'

    ts = str(int(time.time()))

    batch_jobs_dir = args.b_dir
    runs_parent_dir = args.r_dir
    kwargs = "|".join(args.kwargs)

    if kwargs:
        if args.reverse:
            filter_func = lambda _dir: not re.search(kwargs, _dir)
            filter_type = 'inc'
        else:
            filter_func = lambda _dir: re.search(kwargs, _dir)
            filter_type = 'exec'
    else:
        filter_func = lambda _dir: False
        filter_type = ''

    # ORDER BY MODEL
    template_lines = get_template_job_lines() + ['\nNUM_DONE=0\n']

    def discarding_per_filter(_dirs):
        filtered = []
        for _dir in _dirs:
            if filter_func(_dir):
                print(f'Discarding {_dir}')
                continue
            filtered.append(_dir)
        return filtered

    filtered_models_runs =\
        discarding_per_filter(os.listdir(runs_parent_dir))
    filtered_models_runs.sort()

    completed_preds = []
    if args.ignore_completed:
        completed_preds = get_completed_predictions()

    def is_completed_before(_dir, task_name):
        for run_name_p, task_name_p in completed_preds:
            if _dir == run_name_p and task_name == task_name_p:
                return True
        return False
    
    num_ignored = 0
    num_included = 0
    runs = []
    for _dir in filtered_models_runs:
        for task_id in args.task_ids:
            task_name = _TASK_NAMES[task_id]
            if is_completed_before(_dir, task_name):
                print(f'Discarding completeed {_dir} on {task_name}')
                num_ignored += 1
                continue
            runs.append((_dir, task_id, task_name))
            num_included += 1
    print('Ignored in total from previously completed', num_ignored)
    print('Num of included', num_included)
    runs.sort(key=lambda x: x[0])

    runs_batches = np.array_split(runs, args.split)
    for batch_i, runs in enumerate(runs_batches):
        output_filename =\
            f"run_GLUE_b{batch_i+1}of{args.split}" +\
            f"__{filter_type}_" +\
            f"{kwargs.replace('|','-')}__{ts}.job"
        write_job(output_filename, template_lines, runs)
