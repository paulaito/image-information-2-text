from visual_info_extractor.main import main
import argparse

parser = argparse.ArgumentParser(description="Benchmark entrypoint")
parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
parser.add_argument("--download", action="store_true", help="Download datasets before running")

args = parser.parse_args()

main(**vars(args))