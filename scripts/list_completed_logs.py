import os, re

if __name__ == "__main__":
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

    completed_preds = get_completed_predictions()
    completed_preds.sort(key=lambda x: x[0])
    breakpoint()

    