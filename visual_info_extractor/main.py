from visual_info_extractor.benchmark import Benchmark
from visual_info_extractor.evaluator import Evaluator

def main(**kwargs):
    config_path = kwargs.get("config_path")
    download = kwargs.get("download", False)
    is_eval = kwargs.get("is_eval", False)

    if is_eval:
        eval = Evaluator(config_path=config_path)
        eval.run()
    else:
        bench = Benchmark(config_path=config_path, download=download)
        bench.run()