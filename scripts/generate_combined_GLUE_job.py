import os
import re
    
if __name__ == "__main__":

    batch_jobs_dir = 'batch_jobs'
    output_filename = "run_GLUE_combined.job"

    kwargs_except = ["finetuned"]
    exceptions = "|".join(kwargs_except)
    
    runs_parent_dir = "GLUE/data/runs"
    
    with open(os.path.join(batch_jobs_dir, output_filename), mode="w+") as f:
        f.writelines([
            'source $PWD/init_env_on_carc_with_gpu.sh &&\\', '\n',
            'echo "done init env"', '\n\n'
        ])
    
        for model_run_dir in os.listdir(runs_parent_dir):
            if re.search(exceptions, model_run_dir):
                print(f'Discarding {model_run_dir}')
                continue

            f.writelines([
                'source activate $ENV_NAME &&\\', '\n',
                f'sh {os.path.join(runs_parent_dir, model_run_dir, "run.sh")} &&\\', '\n',
                f'echo "done {model_run_dir}"', '\n\n'
            ])
            
        f.write('echo "done with all scripts"')
        
