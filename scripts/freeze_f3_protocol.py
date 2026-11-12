"""Fase 3 — congela el protocolo de evaluación ANTES de tocar Fontibón test.

Escribe outputs/metrics/F3_frozen_protocol.json con: modelo final elegido
solo con validación, checkpoints (con SHA-256), métricas y umbrales, y los
criterios de decisión ya fijados en la Primera Entrega. No carga datos de
test, no entrena y no evalúa nada.

Uso: python scripts/freeze_f3_protocol.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import OUTPUTS_DIR

SEEDS = {"A": [42, 123, 2026], "B": [42, 123], "C": [42, 123, 2026]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    out_path = OUTPUTS_DIR / "metrics" / "F3_frozen_protocol.json"
    if out_path.exists():
        print(f"REFUSING TO OVERWRITE: {out_path} ya existe (el protocolo se congela una sola vez).")
        sys.exit(1)

    runs = {}
    for cfg, seeds in SEEDS.items():
        for seed in seeds:
            name = f"transfer_{cfg}_seed{seed}"
            summary = json.loads((OUTPUTS_DIR / "metrics" / f"{name}_summary.json").read_text(encoding="utf-8"))
            assert summary["fontibon_test_used"] is False, name
            ckpt = OUTPUTS_DIR / "checkpoints" / name / "best.weights.h5"
            runs[name] = {
                "config": cfg,
                "seed": seed,
                "checkpoint": str(ckpt.relative_to(OUTPUTS_DIR.parent)).replace("\\", "/"),
                "checkpoint_sha256": sha256(ckpt),
                "best_epoch": summary["best_epoch"],
                "validation_metrics_ug_m3": summary["validation_metrics_ug_m3"],
            }

    val_rmse = {
        cfg: [runs[f"transfer_{cfg}_seed{s}"]["validation_metrics_ug_m3"]["RMSE"] for s in seeds]
        for cfg, seeds in SEEDS.items()
    }
    val_rmse_mean = {cfg: sum(v) / len(v) for cfg, v in val_rmse.items()}
    final_cfg = min(val_rmse_mean, key=val_rmse_mean.get)
    assert final_cfg == "A", val_rmse_mean

    protocol = {
        "note": (
            "Congelado ANTES de construir o evaluar Fontibón test. Todo lo que sigue se fijó "
            "con información de entrenamiento y validación únicamente. Ningún modelo, semilla, "
            "umbral o hiperparámetro se modificará después de ver test."
        ),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "fontibon_test_touched_before_freeze": False,
        "final_model": {
            "config": final_cfg,
            "description": "STN entrenada desde cero en Fontibón (sin pesos de Beijing)",
            "seeds": SEEDS[final_cfg],
            "selection_rule": (
                "Menor RMSE medio de validación entre las configuraciones con todas sus "
                "corridas completas (A y C con 3 semillas, B con 2). Criterio fijado en la "
                "Primera Entrega (RMSE como métrica principal)."
            ),
            "validation_rmse_mean_by_config": val_rmse_mean,
            "report_as": "media ± desviación estándar muestral (n-1) de las 3 semillas; ninguna semilla se elige como 'la mejor'",
        },
        "descriptive_comparisons_on_test": {
            "B": "semillas 42 y 123 (B2026 interrumpida, excluida como en la Fase 2)",
            "C": "semillas 42, 123 y 2026",
            "purpose": "contrastar la transferencia en test; NO se usan para elegir el modelo final",
            "not_evaluated": {
                "D": "solo piloto de integración, sin entrenamiento completo (decisión: no se completa)",
                "B2026": "entrenamiento interrumpido en la época 83 (decisión: no se completa)",
            },
        },
        "reference_models_on_test": {
            "ridge": {
                "definition": "make_pipeline(StandardScaler(), Ridge(alpha=1.0)) sobre ventanas aplanadas, reajustado exactamente como en la Primera Entrega",
                "must_reproduce_test_rmse": 7.063201147985151,
                "note": "su resultado de test ya se reportó en la Primera Entrega; se recalcula solo para tener predicciones por ventana",
            },
            "persistence": {
                "definition": "PM2.5(t+1) = PM2.5(t) observado en la última hora de la ventana",
                "primera_entrega_test_rmse": 8.232,
            },
        },
        "checkpoint_integrity": {
            "runs": runs,
            "rule": "Antes de evaluar test, cada checkpoint debe reproducir su RMSE de validación de la Fase 2 (tolerancia 1e-3 µg/m3) y conservar su SHA-256.",
        },
        "data_protocol": {
            "pipeline": "src.data.fontibon.prepare_fontibon_stn(include_test=True)",
            "features": ["pm25", "temp", "dew"],
            "expected_test_windows": 2574,
            "scaler": "MinMax de Fontibón ajustado solo con train (el mismo de la Fase 2)",
        },
        "metrics": {
            "primary": "RMSE (µg/m3)",
            "secondary": ["MAE", "R2", "sesgo medio (pred - real)"],
            "high_episodes": {
                "primary_threshold_ug_m3": 40.0,
                "secondary_threshold_ug_m3": 37.0,
                "note": "40 es el umbral exploratorio de la Primera Entrega (no normativo); 37 se reporta también, como en el baseline",
                "metrics": ["RMSE_high", "MAE_high", "Bias_high", "n_high"],
            },
        },
        "decision_criteria_from_primera_entrega": {
            "depth_justified": {
                "text": (
                    "La profundidad está justificada si la STN logra una reducción consistente del "
                    "RMSE respecto a la línea base Ridge (7,063 µg/m3) y mejora el comportamiento "
                    "durante episodios de concentración elevada."
                ),
                "operationalization": {
                    "rmse_consistent": "las 3 semillas de A tienen RMSE de test < 7,063",
                    "high_episodes_improved": "RMSE_high medio de A (umbral 40) < RMSE_high de Ridge en test (17,868)",
                    "verdict": "se justifica solo si se cumplen ambas condiciones",
                },
            },
            "transfer_useful": {
                "text": (
                    "La transferencia desde Beijing se considerará útil únicamente si la STN "
                    "preentrenada supera de manera consistente a la misma arquitectura entrenada "
                    "directamente sobre Fontibón."
                ),
                "operationalization": "B o C con RMSE de test menor que A en cada semilla emparejada y en la media",
            },
        },
        "calibration_protocol": {
            "method": "intervalos conformales split: cuantil del error absoluto en VALIDACIÓN de cada semilla de A",
            "nominal_coverage": [0.80, 0.90],
            "evaluated_on": "cobertura empírica en test, global y en episodios >= 40 µg/m3",
        },
        "post_hoc_analyses": {
            "scope": "explicabilidad (atención, saliencia, casos de respuesta conocida) y análisis de fallas",
            "rule": "son descriptivos y se hacen sobre el modelo final ya congelado; no cambian el modelo ni la selección",
        },
        "known_before_freeze": (
            "Se calcularon métricas de episodios altos de A/B/C solo en VALIDACIÓN, a partir de "
            "predicciones ya guardadas en la Fase 2 (A: RMSE_high medio 17,88 frente a Ridge 17,85)."
        ),
        "training_performed": False,
    }

    out_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Protocolo congelado:", out_path)
    print("Modelo final:", final_cfg, "| RMSE medio de validación por config:",
          {k: round(v, 3) for k, v in val_rmse_mean.items()})


if __name__ == "__main__":
    main()
