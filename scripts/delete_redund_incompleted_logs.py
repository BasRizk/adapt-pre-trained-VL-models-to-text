import os, re
import shutil
if __name__ == "__main__":
    def delete_directory(_dir):
        dst = "GLUE/data/logs_deleted"
        if not os.path.exists(dst):
            os.mkdir(dst)
        shutil.move(_dir, dst)
        # breakpoint()

    def get_completed_predictions():
        finished_preds = set()
        for root, dirs, files in os.walk('GLUE/data/logs', topdown=False):
            prediction_file_found = False
            for name in files:
                if name.startswith('predict_results'):
                    task = re.match(r'predict_results_(.+).txt', name).groups()[0]
                    model = root.split('/')[3]
                    # for bert-base-uncased
                    model = model.replace('-uncased', '')
                    if not ((model, task) in finished_preds):
                        finished_preds.add((model, task))
                        prediction_file_found = True
            if not prediction_file_found and len(root.split('/')) == 5:
                delete_directory(root)
                print(f'Deleting incompleted directory: {root}')
        return list(finished_preds)

    completed_preds = get_completed_predictions()
    completed_preds.sort(key=lambda x: x[0])
    breakpoint()

    