.PHONY: setup download baseline train eval infer test quickstart

setup:
	pip install -r requirements.txt

quickstart:
	python examples/quickstart.py

download:
	python scripts/download_triplet.py --date 2025-06-01 --hour 12

baseline:
	python scripts/make_baselines.py

train:
	python scripts/train_refiner.py --epochs 3 --patch-size 128

eval:
	python scripts/evaluate.py

test:
	python -m pytest tests/ -v

infer:
	python scripts/infer.py --t0 frame_t0.npy --t2 frame_t2.npy
