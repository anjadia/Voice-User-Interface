import sys
import time
import argparse
from pathlib import Path


from ruamel.yaml import YAML

from tools.gmm import GMM
from tools.data_loader import DataLoader

from preprocesing.preprocesing import Preprocessor

import warnings
warnings.filterwarnings("ignore")


def process_asr(data_path, csv_path, config_path):

  dataloader = DataLoader(data_path)

  train, test = dataloader.split_to_train_test()

  with config_path.open() as f:
    config = YAML(typ="safe").load(f)

    preprocessor = Preprocessor(config["preprocessing"])

    gmm_processor = GMM(train, test, config["asr"], preprocessor)

    gmm_processor.prepare_models()

    asr_results = gmm_processor.get_prediction_set(csv_path)

    print(asr_results)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="ASR"
  )

  parser.add_argument(
      "data_path", type=Path, help="Path to input dir/file.",
  )

  parser.add_argument(
      "-c",
      "--csv_results",
      type=Path,
      default=Path(__file__).absolute().parents[1] / "ASR_test_results.csv",
      help="Path to the output csv with prediction results.",
      required=False,
  )
  parser.add_argument(
      "-cf",
      "--config_file",
      type=Path,
      default=Path(__file__).absolute().parents[1] / "config.yaml",
      help="Path to config file.",
  )

  args = parser.parse_args()

  if not args.config_file.exists():
    raise FileNotFoundError(args.config_file)

  if not args.data_path.is_dir():
    raise FileNotFoundError(args.data_path)

  process_asr(args.data_path, args.csv_results, args.config_file)

#wywołanie python asr.py <ścieżka do folderu z całym audio>
# przykład: python asr.py '/home/anjadia/VOI/data'