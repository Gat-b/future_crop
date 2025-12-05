reinstall_package:
	@pip uninstall -y future_crop || true
	@pip install -e .

preprocess_explo_wheat:
	python -m future_crop.ml_logic.preprocessing