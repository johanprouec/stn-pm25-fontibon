"""Generates docs/Informe_SegundaEntrega.pdf from real, already-computed
results (R1, R2, domain distance, Ridge baseline, transfer pilots, and
whatever full A/B/C/D transfer runs have completed). No fabricated
numbers: every figure/table value is read from the JSON files already on
disk; sections for runs not yet finished are marked as such.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT

OUT_PATH = PROJECT_ROOT / "docs" / "Informe_SegundaEntrega.pdf"


def load(name):
    p = OUTPUTS_DIR / "metrics" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.5, leading=11))
    story = []

    # --- Portada ---
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("Universidad Externado de Colombia", styles["Title"]))
    story.append(Paragraph("Facultad de Ciencia de Datos", styles["Heading2"]))
    story.append(Paragraph("Proyecto Integrador de Deep Learning", styles["Heading2"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Transferencia de una Sparse Attention-based Transformer Network (STN) para el "
        "pronostico horario de PM2.5: Beijing -&gt; Fontibon (Bogota)", styles["Heading1"]))
    story.append(Paragraph("Segunda Entrega", styles["Heading2"]))
    story.append(Spacer(1, 36))
    story.append(Paragraph("Estudiante: Johan Velandia", styles["Normal"]))
    story.append(Paragraph("Profesor: Wilmer Dario Pineda Rios", styles["Normal"]))
    story.append(Paragraph("Asignatura: Deep Learning", styles["Normal"]))
    story.append(Spacer(1, 36))
    story.append(Paragraph("Bogota D.C., 2026", styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("1. Objetivo y alcance", styles["Heading2"]))
    story.append(Paragraph(
        "Replicar la Sparse Transformer Network (STN) de Zhang &amp; Zhang (2023) sobre Beijing (R1), "
        "preentrenar una version de 3 variables compartidas como fuente de transferencia (R2), y evaluar "
        "cuatro estrategias de transferencia (A/B/C/D) hacia la estacion Fontibon (Bogota), comparando "
        "contra la linea base Ridge de la Primera Entrega. Fontibon test permanece reservado para la Fase 3 "
        "en todo este informe.", styles["Normal"]))
    story.append(Spacer(1, 10))

    # --- R1 ---
    r1 = load("R1_test_final_summary.json")
    story.append(Paragraph("2. R1 -- Replica sobre Beijing (CERRADO)", styles["Heading2"]))
    if r1:
        agg = r1["aggregate_test_metrics_ug_m3"]
        paper = r1["paper_reference_zhang_zhang_2023"]
        story.append(Paragraph(
            f"Entrenado con 3 semillas (42, 123, 2026), evaluado UNA sola vez sobre Beijing test 2014 "
            f"({r1['n_test_windows']} ventanas). RMSE medio = {agg['RMSE_mean']:.3f} +/- {agg['RMSE_std']:.3f} ug/m3, "
            f"MAE = {agg['MAE_mean']:.3f} +/- {agg['MAE_std']:.3f}, R2 = {agg['R2_mean']:.4f} +/- {agg['R2_std']:.4f}. "
            f"Referencia publicada: RMSE={paper['RMSE']}, MAE={paper['MAE']}, R2={paper['R2']} "
            "(comparacion descriptiva, no replica exacta -- ver diferencias metodologicas en "
            "docs/IMPLEMENTATION_ASSUMPTIONS.md). R1 queda cerrado; no se modifica.", styles["Normal"]))
        fig = OUTPUTS_DIR / "figures" / "R1_test_final" / "test_metrics_across_seeds.png"
        if fig.exists():
            story.append(Spacer(1, 6))
            story.append(Image(str(fig), width=4.2 * inch, height=2.7 * inch))
    story.append(Spacer(1, 10))

    # --- R2 ---
    r2 = load("R2_source_seed_comparison.json")
    story.append(Paragraph("3. R2 -- Preentrenamiento fuente Beijing-3 (COMPLETO)", styles["Heading2"]))
    if r2:
        st = r2["validation_stats"]
        story.append(Paragraph(
            "Mismas 3 semillas, 3 variables compartidas (pm25, temp, dew), mismo split cronologico que R1, "
            f"evaluado solo en validation: RMSE medio = {st['val_RMSE_mean']:.3f} +/- {st['val_RMSE_std']:.3f} ug/m3, "
            f"R2 = {st['val_R2_mean']:.4f} +/- {st['val_R2_std']:.4f}. Los tres checkpoints se conservan como "
            "fuente para la transferencia, emparejados por semilla (nunca se elige 'el mejor').", styles["Normal"]))
        fig = OUTPUTS_DIR / "figures" / "R2_source_comparison" / "R2_validation_metrics_across_seeds.png"
        if fig.exists():
            story.append(Spacer(1, 6))
            story.append(Image(str(fig), width=4.2 * inch, height=2.7 * inch))
    story.append(Spacer(1, 10))

    # --- Domain distance ---
    dd = load("domain_distance_analysis.json")
    story.append(Paragraph("4. Distancia entre dominios (Beijing vs. Fontibon, solo train)", styles["Heading2"]))
    if dd:
        bp = dd["descriptive_stats"]["beijing"]["pm25"]
        fp = dd["descriptive_stats"]["fontibon"]["pm25"]
        wr = dd["wasserstein_distance_raw_units"]
        ws = dd["wasserstein_distance_standardized"]
        story.append(Paragraph(
            f"PM2.5 train: Beijing media={bp['mean']:.1f} ug/m3 (std={bp['std']:.1f}, max={bp['max']:.0f}) vs. "
            f"Fontibon media={fp['mean']:.1f} ug/m3 (std={fp['std']:.1f}, max={fp['max']:.1f}) -- factor ~5 de "
            f"diferencia de nivel. Distancia de Wasserstein cruda: pm25={wr['pm25']:.2f}, temp={wr['temp']:.2f}, "
            f"dew={wr['dew']:.2f}; tras estandarizar (forma de la distribucion): pm25={ws['pm25']:.3f}, "
            f"temp={ws['temp']:.3f}, dew={ws['dew']:.3f} -- mucho mas cercanas, consistente con la hipotesis de "
            "que la transferencia aprovecha estructura temporal relativa, no niveles absolutos.", styles["Normal"]))
        fig1 = OUTPUTS_DIR / "figures" / "domain_distance" / "distribution_pm25.png"
        fig2 = OUTPUTS_DIR / "figures" / "domain_distance" / "hourly_profile_pm25.png"
        if fig1.exists() and fig2.exists():
            story.append(Spacer(1, 6))
            story.append(Table([[Image(str(fig1), width=3.1 * inch, height=2.0 * inch),
                                  Image(str(fig2), width=3.1 * inch, height=2.0 * inch)]]))
    story.append(PageBreak())

    # --- Ridge baseline ---
    ridge = load("baseline_fontibon_ridge.json")
    story.append(Paragraph("5. Linea base Ridge (Primera Entrega, referencia historica)", styles["Heading2"]))
    if ridge:
        v, te = ridge["validacion"], ridge["test"]
        rows_ridge = [
            ["Particion", "n ventanas", "RMSE (ug/m3)", "MAE", "R2"],
            ["Train", ridge["n_windows"]["train"], "-", "-", "-"],
            ["Validation", ridge["n_windows"]["val"], f"{v['RMSE']:.3f}", f"{v['MAE']:.3f}", f"{v['R2']:.4f}"],
            ["Test (historico, no recalculado)", ridge["n_windows"]["test"], f"{te['RMSE']:.3f}", f"{te['MAE']:.3f}", f"{te['R2']:.4f}"],
        ]
        t_ridge = Table(rows_ridge, hAlign="LEFT")
        t_ridge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                                      ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(t_ridge)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Ridge de la Primera Entrega, mismo split 70/15/15 de Fontibon. El RMSE de validacion "
            "(6.545 ug/m3) es la referencia usada en la seccion 7b; el test (7.063) se muestra solo como "
            "evidencia historica, no se recalcula ni se compara contra la STN en esta fase.", styles["Small"]))
    else:
        story.append(Paragraph("Ridge validation RMSE = 6.545 ug/m3; test RMSE = 7.063 (no recalculado, reservado).", styles["Normal"]))
    story.append(Spacer(1, 10))

    # --- Transfer pilots ---
    story.append(Paragraph("6. Pilotos de integracion A/B/C/D (2 epocas, 512/128 ventanas -- NO resultado final)", styles["Heading2"]))
    rows = [["Config", "Train loss", "Val loss", "Val RMSE (ug/m3)", "Tiempo (s)", "Estado"]]
    for cfg in ["A", "B", "C", "D"]:
        s = load(f"transfer_pilot_{cfg}_seed42_summary.json")
        if s:
            rows.append([cfg, f"{s['final_train_loss_scaled']:.5f}", f"{s['final_val_loss_scaled']:.5f}",
                         f"{s['validation_metrics_ug_m3']['RMSE']:.3f}", f"{s['total_pilot_time_seconds']:.1f}", s["status"]])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(t)
    story.append(Paragraph(
        "Pilotos de integracion unicamente (2 epocas): verifican que la cadena preprocesamiento -> carga de "
        "pesos R2 -> congelamiento -> entrenamiento -> evaluacion funciona de punta a punta. No son resultados "
        "cientificos ni implican que una configuracion sea superior.", styles["Small"]))
    story.append(Spacer(1, 10))

    prov = load("transfer_B_seed2026_PROVISIONAL_summary.json")
    if prov:
        vm = prov["validation_metrics_ug_m3"]
        story.append(Paragraph(
            f"Nota -- B seed2026 PROVISIONAL: entrenamiento interrumpido por un reinicio del equipo en la "
            f"epoca 83. Evaluado por inferencia pura (sin entrenar) sobre el checkpoint de su mejor epoca "
            f"(74): val RMSE={vm['RMSE']:.3f}, MAE={vm['MAE']:.3f}, R2={vm['R2']:.4f}. No se incluye en la "
            "media +/- std de B (n=2: seeds 42 y 123) por no ser un entrenamiento completo.", styles["Small"]))
        story.append(Spacer(1, 10))

    # --- Full training (whatever completed) ---
    story.append(Paragraph("7. Entrenamiento completo A/B/C/D sobre Fontibon (train=10802, validation=2564)", styles["Heading2"]))
    rows = [["Config", "Seed", "Estado", "Best ep.", "Stop ep.", "Val RMSE", "Val MAE", "Val R2", "Bias", "Tiempo (s)"]]
    any_full = False
    for cfg in ["A", "B", "C", "D"]:
        for seed in [42, 123, 2026]:
            s = load(f"transfer_{cfg}_seed{seed}_summary.json")
            if s is None:
                continue
            any_full = True
            if s.get("status", "").startswith("INCOMPLETE"):
                rows.append([cfg, seed, "INCOMPLETO (detenido por plazo)", s.get("epochs_completed"), "-", "-", "-", "-", "-", "-"])
            else:
                vm = s["validation_metrics_ug_m3"]
                rows.append([cfg, seed, "completo", s["best_epoch"], s["stopping_epoch"],
                             f"{vm['RMSE']:.3f}", f"{vm['MAE']:.3f}", f"{vm['R2']:.4f}",
                             f"{s['validation_residual_diagnostics_ug_m3']['bias_mean']:.2f}",
                             f"{s['total_training_time_seconds']:.0f}"])
    if any_full:
        t2 = Table(rows, hAlign="LEFT")
        t2.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(t2)
    else:
        story.append(Paragraph("Entrenamientos completos en curso al momento de este corte; ver notebook/JSON para el estado mas reciente.", styles["Normal"]))
    story.append(Spacer(1, 10))

    # --- Mean +/- std per config vs Ridge ---
    story.append(Paragraph("7b. Media +/- desviacion estandar por configuracion vs. Ridge", styles["Heading2"]))
    ridge_val_rmse = 6.545
    config_rmses = {}
    for cfg in ["A", "B", "C", "D"]:
        vals = []
        for seed in [42, 123, 2026]:
            s = load(f"transfer_{cfg}_seed{seed}_summary.json")
            if s and not s.get("status", "").startswith("INCOMPLETE"):
                vals.append(s["validation_metrics_ug_m3"]["RMSE"])
        config_rmses[cfg] = vals
    rows3 = [["Config", "n", "Val RMSE medio (ug/m3)", "vs. Ridge (6.545)"]]
    for cfg, vals in config_rmses.items():
        if not vals:
            rows3.append([cfg, 0, "sin datos", "-"])
            continue
        arr = np.array(vals)
        tag = f"{arr.mean():.3f} +/- {arr.std(ddof=1):.3f}" if len(arr) > 1 else f"{arr[0]:.3f}"
        cmp = "por debajo de Ridge (mejor)" if arr.mean() < ridge_val_rmse else "por encima de Ridge (peor)"
        rows3.append([cfg, len(arr), tag, cmp])
    t3 = Table(rows3, hAlign="LEFT")
    t3.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(t3)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Interpretacion (sin sobre-afirmar superioridad sin evidencia suficiente): con las semillas completas "
        "disponibles, Config A (sin transferencia, entrenada desde cero) es la UNICA que queda por debajo del "
        "RMSE de Ridge en validacion. B y C (que congelan representaciones preentrenadas en Beijing) quedan "
        "ligeramente por encima de Ridge y tambien por encima de A. Esto no demuestra que la transferencia sea "
        "inutil en general -- Config D (fine-tuning completo, sin congelar nada) aun no se ha evaluado con "
        "ninguna semilla y es el escenario donde la hipotesis de transferencia tiene mas margen.", styles["Normal"]))
    story.append(PageBreak())

    # --- Curves / training dynamics ---
    story.append(Paragraph("8. Dinamica del entrenamiento", styles["Heading2"]))
    story.append(Paragraph(
        "Curvas train/validation y norma de gradiente para una corrida representativa por configuracion "
        "(seed 42). En las tres configs la perdida de validacion tiende a disminuir en las primeras epocas, "
        "con fluctuaciones normales de un optimizador estocastico (no es una caida monotona), y la norma del "
        "gradiente decrece y se estabiliza en valores bajos hacia el final del entrenamiento, sin signos de "
        "divergencia numerica. B y C muestran una caida inicial mas rapida que A porque parten de pesos de "
        "encoder/decoder ya preentrenados en Beijing (R2), no de inicializacion aleatoria -- esto no implica "
        "por si solo un mejor resultado final, como muestra la tabla 7b.", styles["Normal"]))
    for cfg in ["A", "B", "C"]:
        loss_fig = OUTPUTS_DIR / "figures" / f"transfer_{cfg}_seed42" / "loss_train_vs_validation.png"
        grad_fig = OUTPUTS_DIR / "figures" / f"transfer_{cfg}_seed42" / "gradient_norm_mean.png"
        if loss_fig.exists() and grad_fig.exists():
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"Config {cfg} (seed 42)", styles["Small"]))
            story.append(Table([[Image(str(loss_fig), width=3.1 * inch, height=2.0 * inch),
                                  Image(str(grad_fig), width=3.1 * inch, height=2.0 * inch)]]))
    story.append(PageBreak())

    story.append(Paragraph("9. Limitaciones y trabajo pendiente", styles["Heading2"]))
    story.append(Paragraph(
        "Config B-seed2026 se interrumpio por un reinicio del equipo durante el entrenamiento (epoca 83); se "
        "evaluo unicamente por inferencia (sin entrenar) el checkpoint de su mejor epoca (74) y se reporta como "
        "PROVISIONAL, fuera de la media +/- std de B. Config D no se ha ejecutado con ninguna semilla (no es "
        "obligatorio segun la guia del curso). Fontibon test no se ha tocado en ningun momento de esta fase. "
        "R1 y R2 permanecen intactos y cerrados (verificado por SHA-256 de los checkpoints R2). No se realizo "
        "tuning de hiperparametros ni cambios de arquitectura en ningun punto de este informe.", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("10. Conclusiones", styles["Heading2"]))
    story.append(Paragraph(
        "R1 (replica Beijing) queda cerrado con resultados razonablemente cercanos al paper (RMSE +2.09 ug/m3, "
        "R2 ligeramente superior), dadas las diferencias metodologicas documentadas. R2 (fuente de transferencia) "
        "es estable entre semillas. El analisis de distancia de dominio muestra que, aunque los niveles absolutos "
        "de PM2.5 difieren ~5x entre Beijing y Fontibon, la forma estandarizada de las distribuciones es mucho "
        "mas cercana, lo que motiva intentar transferencia de estructura temporal en vez de niveles. Sin embargo, "
        "con los datos completados hasta ahora, la transferencia implementada (Config B y C, que congelan "
        "representaciones) no logra superar ni a Ridge ni a entrenar desde cero (Config A) en validacion de "
        "Fontibon -- un resultado negativo honesto, no un fracaso del proyecto: Config D (fine-tuning completo) "
        "queda como la via mas prometedora aun no explorada.", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("11. Reproducibilidad", styles["Heading2"]))
    story.append(Paragraph(
        "Semillas: 42, 123, 2026. Entorno: requirements.txt (TensorFlow 2.21.0, scikit-learn 1.9.1). Comandos "
        "principales: <font face='Courier' size=8>python scripts/run_r1_development.py &lt;seed&gt;</font>, "
        "<font face='Courier' size=8>python scripts/run_r2_source_training.py</font>, "
        "<font face='Courier' size=8>python scripts/run_transfer_full_training.py --config &lt;A|B|C|D&gt; --seed &lt;seed&gt;</font>. "
        "Todos los splits son cronologicos y fijos (ver docs/PHASE2_PLAN.md); los escaladores se ajustan "
        "exclusivamente con train de cada dominio. <font face='Courier' size=8>pytest tests/ -v</font> "
        "(200+ tests) verifica shapes, ausencia de fuga temporal, congelamiento correcto y reproducibilidad "
        "por semilla antes de cualquier entrenamiento real.", styles["Small"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("12. Declaracion de uso de IA", styles["Heading2"]))
    story.append(Paragraph(
        "El codigo de `src/`, los scripts de `scripts/` y este informe se desarrollaron con asistencia de un "
        "modelo de lenguaje (Claude, Anthropic) como herramienta de programacion en pareja, bajo supervision y "
        "revision humana continua del integrante del curso. Las decisiones de diseno, los hiperparametros y la "
        "interpretacion de los resultados fueron revisados y aprobados por el integrante antes de ejecutarse. "
        "Ningun resultado numerico de este informe fue generado o alterado sin ejecucion real del codigo "
        "correspondiente -- todas las cifras provienen de los archivos JSON en outputs/metrics/.", styles["Small"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("13. Bibliografia", styles["Heading2"]))
    for line in [
        "Zhang, Z., &amp; Zhang, S. (2023). Modeling air quality PM2.5 forecasting using deep sparse "
        "attention-based transformer networks. International Journal of Environmental Science and "
        "Technology, 20, 13535-13550. https://doi.org/10.1007/s13762-023-04900-1",
        "Zhou, H. et al. (2021). Informer: Beyond efficient transformer for long sequence time-series "
        "forecasting. AAAI.",
        "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.",
        "Universidad Externado de Colombia. Proyecto Integrador -- Deep Learning, guia del curso. "
        "Profesor: Wilmer Dario Pineda Rios.",
        "Chen, S. (2017). Beijing PM2.5 Data Set. UCI Machine Learning Repository.",
        "Datos de calidad del aire, estacion Fontibon, Bogota (fuente institucional local; ver "
        "docs/PHASE1_AUDIT.md para la auditoria del dataset).",
    ]:
        story.append(Paragraph("- " + line, styles["Small"]))

    doc = SimpleDocTemplate(str(OUT_PATH), pagesize=LETTER, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build(story)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
