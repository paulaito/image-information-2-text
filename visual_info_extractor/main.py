from visual_info_extractor.benchmark import Benchmark

def main(**kwargs):
    config_path = kwargs.get("config_path")
    download = kwargs.get("download", False)

    bench = Benchmark(config_path=config_path, download=download)
    bench.run()