# Screenshots

Azure ML studio views (ml.azure.com) from the October 2026 runs:

1. `pipeline_run.png` - Jobs -> credit-default -> latest run: the 4-step graph, all green.
2. `evaluate_metrics.png` - the *3 - evaluate* step -> Metrics (test AUC, Gini, KS, ECE ...).
3. `evaluate_images.png` - the *3 - evaluate* step -> Images (backtest, calibration, SHAP).
4. `model_registered.png` - Models -> credit-default-model -> latest version (tags and artifacts incl. MODEL_CARD.md).
5. `data_asset.png` - Data -> credit-default-raw (versioned, sha256 tag).
6. `compute_cluster.png` - Compute -> Compute clusters -> cpu-cluster (min 0 nodes).
7. `batch_monitoring.png` - the downturn batch run with the drift alert.

The online endpoint was deleted on 2 November 2026; its real response is in `deploy/sample_response.json`.
