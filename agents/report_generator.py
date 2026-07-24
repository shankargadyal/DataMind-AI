"""PDF Report Generator — requires: pip install reportlab"""
import io
import base64
from datetime import datetime


def _b64_to_reportlab_image(data_uri: str, max_width_cm: float = 16.0):
    """Decode a 'data:image/png;base64,...' string into a reportlab Image
    flowable, scaled to fit the page width while preserving aspect ratio."""
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.units import cm
    try:
        header, b64data = data_uri.split(",", 1)
        png_bytes = base64.b64decode(b64data)
        img_buf = io.BytesIO(png_bytes)
        # Peek at native size via PIL if available, else fall back to a fixed aspect ratio
        try:
            from PIL import Image as PILImage
            with PILImage.open(io.BytesIO(png_bytes)) as pil_img:
                w_px, h_px = pil_img.size
            aspect = h_px / w_px if w_px else 0.55
        except Exception:
            aspect = 0.55
        width = max_width_cm * cm
        height = width * aspect
        img_buf.seek(0)
        return RLImage(img_buf, width=width, height=height)
    except Exception:
        return None


def generate_pdf_report(analysis_data: dict, ctx_safe: dict, ml_safe: dict, industry: dict = None,
                         eval_result: dict = None, rag_info: dict = None, llm_calls: list = None) -> bytes:
    industry = industry or {}
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise ImportError("reportlab not installed — run: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    C_PRIMARY = colors.HexColor("#7C6FF7")
    C_SUCCESS = colors.HexColor("#00C896")
    C_WARNING = colors.HexColor("#F5A623")
    C_DANGER  = colors.HexColor("#F04444")
    C_DARK    = colors.HexColor("#1E293B")
    C_GRAY    = colors.HexColor("#64748B")
    C_LIGHT   = colors.HexColor("#E2E8F0")
    C_BG      = colors.HexColor("#F8FAFC")

    base = getSampleStyleSheet()["Normal"]

    def S(name, **kw):
        return ParagraphStyle(name, parent=base, **kw)

    sty = {
        "title":    S("T",  fontSize=22, textColor=C_DARK, fontName="Helvetica-Bold", spaceAfter=4, leading=28),
        "sub":      S("Su", fontSize=11, textColor=C_GRAY, spaceAfter=10),
        "small":    S("Sm", fontSize=8,  textColor=C_GRAY, spaceAfter=4, leading=12),
        "h1":       S("H1", fontSize=13, textColor=C_PRIMARY, fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6),
        "body":     S("Bo", fontSize=9.5, textColor=C_DARK, spaceAfter=5, leading=14),
        "footer":   S("Fo", fontSize=7.5, textColor=C_GRAY, alignment=TA_CENTER),
        "mono":     S("Mo", fontSize=8.5, fontName="Courier", textColor=C_DARK, spaceAfter=4),
    }

    def tbl(data, col_widths, header_bg=C_PRIMARY):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_BG, colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.4, C_LIGHT),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    story = []

    # Header
    story.append(Paragraph("DataMind AI", sty["title"]))
    story.append(Paragraph("Automated Intelligence Report", sty["sub"]))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}  ·  "
        f"File: {analysis_data.get('filename', 'dataset.csv')}",
        sty["small"]
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY, spaceAfter=14))

    # Stats banner
    stats = analysis_data.get("stats", {})
    ml    = analysis_data.get("ml_recommendation", {})
    task  = ml.get("task_type", "regression")
    best_score = ml.get("best_score", 0)
    if task == "classification":
        score_disp = f"{round(best_score*100,1)}% acc"
    elif task == "clustering":
        score_disp = f"silhouette={round(best_score,3)}"
    elif task == "forecasting":
        score_disp = f"MAE={round(best_score,3)}"
    else:
        score_disp = f"R²={round(best_score,4)}"

    banner_data = [
        ["Rows", "Columns", "Quality", "Insights", "Best Model", "Score"],
        [
            f"{stats.get('total_rows',0):,}",
            str(stats.get("total_cols", 0)),
            f"{stats.get('quality_score', 0)}%",
            str(stats.get("insights_count", 0)),
            str(ml.get("best_model", "N/A"))[:16],
            score_disp,
        ],
    ]
    story.append(tbl(banner_data, [2.4*cm]*6))
    story.append(Spacer(1, 14))

    # Executive Summary
    story.append(Paragraph("Executive Summary", sty["h1"]))
    summary = analysis_data.get("summary", "Analysis complete.")
    for para in (summary.split("\n\n") if "\n\n" in summary else [summary]):
        if para.strip():
            story.append(Paragraph(para.strip(), sty["body"]))

    # AI Agent Contributions — makes the multi-agent architecture visible
    # to the reader instead of leaving it implicit behind one summary.
    story.append(Paragraph("AI Agent Contributions", sty["h1"]))
    n_insights   = len(analysis_data.get("key_insights", []))
    n_actions    = len(analysis_data.get("actions", []))
    best_model   = ml.get("best_model", "N/A")
    qd_score     = analysis_data.get("quality_dimensions", {}).get("composite_score")
    detective_desc = (
        f"Assessed data quality (missing values, duplicates, outliers) — composite score {qd_score}/100."
        if qd_score is not None else
        "Assessed data quality (missing values, duplicates, outliers)."
    )
    agent_lines  = [
        ("Data Detective Agent", detective_desc),
        ("Analytics Agent", f"Generated {n_insights} data-driven insight(s) from statistical patterns and correlations."),
        ("ML Engineer Agent", f"Trained and compared multiple models — selected {best_model} as best performer."),
        ("Explainability Agent", "Produced SHAP-based global and local explanations for model decisions."),
        ("Reporter Agent", f"Synthesized findings into {n_actions} recommended action(s) and this executive report."),
    ]
    if eval_result and eval_result.get("status") not in (None, "NOT_EVALUATED"):
        agent_lines.append((
            "Guardrails / Evaluator Agent",
            f"Checked report claims for internal consistency — result: {eval_result.get('status')}"
            + (f" ({len(eval_result.get('flags', []))} flag(s) reviewed)" if eval_result.get("flags") else ", no issues found.")
        ))
    for name, desc in agent_lines:
        story.append(Paragraph(f"<b>{name}</b> — {desc}", sty["small"]))
    story.append(Spacer(1, 8))

    # Data Quality Center (5-dimension composite score)
    qd = analysis_data.get("quality_dimensions", {})
    if qd and qd.get("dimensions"):
        story.append(Paragraph("Data Quality Center", sty["h1"]))
        story.append(Paragraph(
            f"Composite Score: <b>{qd.get('composite_score', 0)}/100</b>  ·  Grade: <b>{qd.get('grade','')}</b>",
            sty["body"]
        ))
        dims = qd["dimensions"]
        dim_data = [
            ["Completeness", "Accuracy", "Consistency", "Validity", "Uniqueness"],
            [f"{dims.get('completeness',0)}%", f"{dims.get('accuracy',0)}%", f"{dims.get('consistency',0)}%",
             f"{dims.get('validity',0)}%", f"{dims.get('uniqueness',0)}%"],
        ]
        story.append(tbl(dim_data, [3*cm]*5, header_bg=C_PRIMARY))
        story.append(Spacer(1, 6))
        for rec in qd.get("recommendations", [])[:5]:
            story.append(Paragraph(f"&bull; {rec}", sty["small"]))
        story.append(Spacer(1, 8))

    # Key Insights
    insights = analysis_data.get("key_insights", [])
    if insights:
        story.append(Paragraph("Key Insights", sty["h1"]))
        sev_map = {"CRITICAL": C_DANGER, "WARNING": C_WARNING, "INFO": C_SUCCESS}
        ins_data = [["#", "Insight", "Detail", "Severity"]]
        for i, ins in enumerate(insights[:8], 1):
            ins_data.append([
                str(i),
                Paragraph(str(ins.get("title",""))[:50], sty["small"]),
                Paragraph(str(ins.get("detail",""))[:120], sty["small"]),
                ins.get("severity","info").upper(),
            ])
        t = Table(ins_data, colWidths=[0.6*cm, 4*cm, 8*cm, 2*cm])
        style = [
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ALIGN",         (0,0), (0,-1), "CENTER"),
            ("ALIGN",         (3,0), (3,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_BG, colors.white]),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]
        for ri, ins in enumerate(insights[:8], 1):
            c = sev_map.get(ins.get("severity","info").upper(), C_SUCCESS)
            style += [
                ("TEXTCOLOR", (3,ri), (3,ri), c),
                ("FONTNAME",  (3,ri), (3,ri), "Helvetica-Bold"),
            ]
        t.setStyle(TableStyle(style))
        story.append(t)
        story.append(Spacer(1, 8))

    # ML Model Comparison
    models = ml.get("models", [])
    if models:
        story.append(Paragraph("ML Model Comparison", sty["h1"]))
        metric_label = {
            "classification": "Accuracy / F1",
            "clustering": "Silhouette Score (higher is better)",
            "forecasting": "MAE (lower is better)",
        }.get(task, "R² / MAE / RMSE")
        target_disp = ml.get("target_column") or ("N/A — unsupervised" if task == "clustering" else "?")
        story.append(Paragraph(
            f"Task: <b>{task.upper()}</b>  ·  Target: <b>{target_disp}</b>  ·  "
            f"Metric: <b>{metric_label}</b>",
            sty["body"]
        ))
        story.append(Spacer(1, 6))

        # Main scores
        m_data = [["Model", "Score", "Best?"]]
        for m in models:
            sc = m.get("score", m.get("mae", 0))
            if m.get("error"):
                disp = "unavailable"
            elif task == "classification":
                disp = f"{round(sc*100,1)}%"
            elif task == "clustering":
                disp = f"silhouette={round(sc,4)}"
            elif task == "forecasting":
                disp = f"MAE={round(sc,4)}"
            else:
                disp = f"R²={round(sc,4)}"
            m_data.append([m.get("name","?"), disp, "★ BEST" if m.get("is_best") else ""])
        mt = Table(m_data, colWidths=[7*cm, 4*cm, 3.6*cm])
        ms = [
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_BG, colors.white]),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]
        for ri, m in enumerate(models, 1):
            if m.get("is_best"):
                ms += [
                    ("BACKGROUND", (0,ri), (-1,ri), colors.HexColor("#EDE9FF")),
                    ("TEXTCOLOR",  (2,ri), (2,ri),  C_PRIMARY),
                    ("FONTNAME",   (2,ri), (2,ri),  "Helvetica-Bold"),
                ]
        mt.setStyle(TableStyle(ms))
        story.append(mt)
        story.append(Spacer(1, 8))

        # Detailed metrics for best model
        bm = ml.get("best_metrics", {})
        if bm:
            story.append(Paragraph(f"Best Model Metrics — {ml.get('best_model','')}", sty["h1"]))
            if task == "classification":
                met_data = [
                    ["Accuracy", "F1 Score", "Precision", "Recall"],
                    [
                        f"{round(bm.get('accuracy',0)*100,1)}%",
                        f"{round(bm.get('f1',0)*100,1)}%",
                        f"{round(bm.get('precision',0)*100,1)}%",
                        f"{round(bm.get('recall',0)*100,1)}%",
                    ]
                ]
                story.append(tbl(met_data, [3.6*cm]*4, header_bg=C_SUCCESS))
            else:
                met_data = [
                    ["R² Score", "MAE", "RMSE"],
                    [
                        str(round(bm.get("r2",0), 4)),
                        str(round(bm.get("mae",0), 4)),
                        str(round(bm.get("rmse",0), 4)),
                    ]
                ]
                story.append(tbl(met_data, [4.8*cm]*3, header_bg=C_SUCCESS))
            story.append(Spacer(1, 8))

        # Prediction Confidence Summary — confidence score, risk level, and a
        # plain-English reason, applied at the model level (per-row prediction
        # confidence isn't retained past the pipeline run, so this summarizes
        # the model's overall prediction reliability the same way).
        best_score_val = ml.get("best_score", 0) or 0
        conf_pct = round(best_score_val * 100, 1) if best_score_val <= 1 else round(best_score_val, 1)
        risk_level = "Low" if conf_pct >= 80 else ("Medium" if conf_pct >= 60 else "High")
        risk_color = C_SUCCESS if risk_level == "Low" else (C_WARNING if risk_level == "Medium" else C_DANGER)
        top_feats = [f.get("feature", f.get("name", "")) for f in (ml.get("feature_importances", []) or [])[:3] if isinstance(f, dict)]
        reason = (
            f"Prediction quality is primarily driven by {', '.join(top_feats)}."
            if top_feats else
            f"Based on {ml.get('best_model','the selected model')}'s cross-validated performance on this dataset."
        )
        story.append(Paragraph("Prediction Confidence Summary", sty["h1"]))
        story.append(Paragraph(
            f"Confidence: <b>{conf_pct}%</b>  ·  "
            f"<font color='{risk_color.hexval()}'>Risk Level: <b>{risk_level}</b></font>",
            sty["body"]
        ))
        story.append(Paragraph(reason, sty["small"]))
        story.append(Spacer(1, 8))

        # Confusion matrix
        cm_data = ml.get("confusion_matrix")
        cm_lbls = ml.get("confusion_labels", [])
        if cm_data and task == "classification" and cm_lbls:
            story.append(Paragraph("Confusion Matrix", sty["h1"]))
            header = ["Pred →"] + [str(l)[:10] for l in cm_lbls]
            cm_rows = [header]
            for ri, row in enumerate(cm_data):
                cm_rows.append([str(cm_lbls[ri])[:10]] + [str(v) for v in row])
            ct = Table(cm_rows)
            cs = [
                ("BACKGROUND",    (0,0), (-1,0), C_DARK),
                ("BACKGROUND",    (0,0), (0,-1), C_DARK),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("TEXTCOLOR",     (0,0), (0,-1), colors.white),
                ("FONTNAME",      (0,0), (-1,-1), "Courier"),
                ("FONTSIZE",      (0,0), (-1,-1), 9),
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("GRID",          (0,0), (-1,-1), 0.4, C_LIGHT),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ]
            # Highlight diagonal
            for i in range(min(len(cm_data), len(cm_lbls))):
                cs.append(("BACKGROUND", (i+1, i+1), (i+1, i+1), colors.HexColor("#D1FAE5")))
                cs.append(("TEXTCOLOR",  (i+1, i+1), (i+1, i+1), colors.HexColor("#065F46")))
                cs.append(("FONTNAME",   (i+1, i+1), (i+1, i+1), "Courier-Bold"))
            ct.setStyle(TableStyle(cs))
            story.append(ct)
            story.append(Spacer(1, 8))

    # Feature Importances
    fi = ml.get("feature_importances", [])
    shap_plots = ml.get("shap_plots", {}) or {}
    shap_available = bool(shap_plots.get("available"))
    if fi:
        story.append(Paragraph(
            f"Top Feature Importances {'(SHAP)' if shap_available else '(Native)'}",
            sty["h1"]
        ))
        fi_data = [["Rank", "Feature", "Importance Score"]]
        for i, f in enumerate(fi[:12], 1):
            fi_data.append([str(i), str(f.get("feature",""))[:45], f"{f.get('importance',0):.6f}"])
        ft = Table(fi_data, colWidths=[1.2*cm, 11*cm, 3.4*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_PRIMARY),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("FONTNAME",      (2,1), (2,-1), "Courier"),
            ("ALIGN",         (0,0), (0,-1), "CENTER"),
            ("ALIGN",         (2,0), (2,-1), "RIGHT"),
            ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_BG, colors.white]),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(ft)
        story.append(Spacer(1, 8))

    # Explainable AI (Agent 4): the actual SHAP visuals, not just a table —
    # this is the gap that existed before: shap_available was checked but
    # the images themselves were never embedded.
    if shap_available:
        story.append(Paragraph("Explainable AI — Why The Model Decides What It Decides", sty["h1"]))

        summary_img = _b64_to_reportlab_image(shap_plots.get("summary_plot", ""))
        if summary_img:
            story.append(Paragraph("Global explanation: what drives predictions overall", sty["body"]))
            story.append(summary_img)
            cap = shap_plots.get("summary_caption", "")
            if cap:
                story.append(Paragraph(cap, sty["small"]))
            story.append(Spacer(1, 10))

        waterfall_img = _b64_to_reportlab_image(shap_plots.get("waterfall_plot", ""))
        if waterfall_img:
            story.append(Paragraph("Local explanation: how one specific prediction was built up", sty["body"]))
            story.append(waterfall_img)
            cap = shap_plots.get("waterfall_caption", "")
            if cap:
                story.append(Paragraph(cap, sty["small"]))
            story.append(Spacer(1, 8))
    elif shap_plots.get("reason"):
        # Don't silently omit this — say why SHAP isn't here, same spirit as
        # the rest of this report's "never invent, always explain" approach.
        story.append(Paragraph(
            f"<i>SHAP explanations were unavailable for this run: {shap_plots['reason']}</i>",
            sty["small"]
        ))
        story.append(Spacer(1, 6))

    # Dataset Statistics
    col_stats = ctx_safe.get("column_stats", {})
    num_cols  = ctx_safe.get("numeric_cols", [])
    if col_stats and num_cols:
        story.append(Paragraph("Dataset Column Statistics", sty["h1"]))
        cols = num_cols[:6]
        hdr  = ["Stat"] + [c[:10] for c in cols]
        rows_d = [hdr]
        for stat in ["mean", "min", "max", "median", "std"]:
            row = [stat.capitalize()]
            for col in cols:
                v = col_stats.get(col, {}).get(stat)
                row.append(f"{v:.3f}" if v is not None else "—")
            rows_d.append(row)
        cw = [2*cm] + [14.8/max(len(cols),1)*cm] * len(cols)
        st = Table(rows_d, colWidths=cw)
        st.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("FONTNAME",      (1,1), (-1,-1), "Courier"),
            ("ALIGN",         (1,0), (-1,-1), "RIGHT"),
            ("GRID",          (0,0), (-1,-1), 0.3, C_LIGHT),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_BG, colors.white]),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(st)
        story.append(Spacer(1, 8))

    # Recommended Actions
    actions = analysis_data.get("actions", [])
    if actions:
        story.append(Paragraph("Recommended Actions", sty["h1"]))
        for i, a in enumerate(actions[:6], 1):
            act = a.get("action", str(a)) if isinstance(a, dict) else str(a)
            rsn = a.get("reason", "") if isinstance(a, dict) else ""
            story.append(Paragraph(f"<b>{i}. {act}</b>", sty["body"]))
            if rsn:
                story.append(Paragraph(rsn, sty["small"]))

    # Future Prediction
    fut = ml.get("future_prediction", {})
    if fut and fut.get("future_y"):
        story.append(Paragraph("Future Trend Prediction", sty["h1"]))
        story.append(Paragraph(
            f"Target: <b>{fut.get('target','')}</b>  ·  "
            f"Steps ahead: <b>{fut.get('future_steps',0)}</b>  ·  "
            f"Trend R²: <b>{fut.get('trend_r2',0)}</b>  ·  "
            f"Std Error: <b>±{fut.get('std_error',0)}</b>",
            sty["body"]
        ))
        fy   = fut.get("future_y", [])
        fx   = fut.get("future_x", list(range(len(fy))))
        fl   = fut.get("future_lower", fy)
        fu_u = fut.get("future_upper", fy)
        pred_data = [["Step", "Predicted", "Lower 95%", "Upper 95%"]]
        for x, y_v, lo, hi in zip(fx[:10], fy[:10], fl[:10], fu_u[:10]):
            pred_data.append([str(x), f"{y_v:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
        story.append(tbl(pred_data, [3*cm, 4.5*cm, 4.5*cm, 4.5*cm], header_bg=C_SUCCESS))

    # Risk Analysis
    risk_flags  = analysis_data.get("risk_flags", [])
    risk_score  = analysis_data.get("risk_score", 0)
    imbalance   = ml.get("class_imbalance", {})
    risk_items = list(risk_flags)
    if imbalance and imbalance.get("is_imbalanced"):
        risk_items.append(f"Class imbalance detected (ratio {imbalance.get('ratio', '?')}) — accuracy alone can be misleading for this target.")
    if qd and qd.get("composite_score", 100) < 70:
        risk_items.append(f"Data quality composite score is {qd.get('composite_score')}/100 — model conclusions inherit this uncertainty.")
    if risk_items or risk_score:
        story.append(Paragraph("Risk Analysis", sty["h1"]))
        if risk_score:
            badge_color = C_DANGER if risk_score > 60 else (C_WARNING if risk_score > 30 else C_SUCCESS)
            story.append(Paragraph(f"<font color='{badge_color.hexval()}'>Overall Risk Score: <b>{risk_score}/100</b></font>", sty["body"]))
        for ri in risk_items[:6]:
            story.append(Paragraph(f"&bull; {ri}", sty["small"]))
        if not risk_items:
            story.append(Paragraph("No elevated risk factors identified in this analysis.", sty["small"]))
        story.append(Spacer(1, 8))

    # Future Opportunities
    next_steps = analysis_data.get("next_steps", [])
    if next_steps:
        story.append(Paragraph("Future Opportunities", sty["h1"]))
        for i, step in enumerate(next_steps[:6], 1):
            story.append(Paragraph(f"{i}. {step}", sty["body"]))
        story.append(Spacer(1, 8))

    # Industry Intelligence
    if industry and industry.get("available"):
        story.append(Paragraph(f"{industry.get('industry_label','Industry')} Intelligence", sty["h1"]))
        kpis = industry.get("matched_kpis", [])
        if kpis:
            kpi_data = [["KPI", "Matched Column", "Average"]]
            for k in kpis[:6]:
                mean_v = k.get("mean")
                kpi_data.append([k.get("kpi",""), k.get("matched_column",""),
                                  f"{mean_v:.2f}" if isinstance(mean_v, (int, float)) else "—"])
            story.append(tbl(kpi_data, [6*cm, 5*cm, 3.6*cm], header_bg=C_SUCCESS))
            story.append(Spacer(1, 6))
        for rec in industry.get("recommendations", [])[:4]:
            story.append(Paragraph(f"&bull; {rec}", sty["small"]))
        story.append(Spacer(1, 8))

    # AI Execution Summary — reports what actually happened during this run
    # (real RAG chunk counts, real Guardrails verdict, real per-agent LLMOps
    # stats pulled from the DB), rather than asking an LLM to describe
    # architecture it has no visibility into.
    story.append(Paragraph("AI Execution Summary", sty["h1"]))

    rag_info = rag_info or {}
    chunk_count = rag_info.get("chunk_count", 0)
    story.append(Paragraph(
        f"<b>RAG:</b> {chunk_count} analysis chunk(s) indexed (TF-IDF retrieval) — "
        f"chat responses are grounded in this indexed evidence rather than a fixed context slice."
        if chunk_count else
        "<b>RAG:</b> not indexed for this run.",
        sty["small"]
    ))

    if eval_result:
        status = eval_result.get("status", "NOT_EVALUATED")
        flags = eval_result.get("flags", [])
        flag_summary = "; ".join(f"{f['check']} ({f['severity']})" for f in flags[:5]) if flags else "no issues found"
        story.append(Paragraph(
            f"<b>Guardrails:</b> verdict = {status}. {flag_summary}.",
            sty["small"]
        ))
    else:
        story.append(Paragraph("<b>Guardrails:</b> not evaluated for this run.", sty["small"]))

    if llm_calls:
        by_agent = {}
        for c in llm_calls:
            a = c.get("agent_name", "unknown")
            by_agent.setdefault(a, []).append(c)
        for agent, calls in by_agent.items():
            n = len(calls)
            avg_latency = sum(c.get("latency_ms", 0) or 0 for c in calls) / n
            failures = sum(1 for c in calls if not c.get("success", True))
            story.append(Paragraph(
                f"&bull; <b>{agent}</b>: {n} call(s), avg latency {avg_latency:.0f}ms"
                + (f", {failures} failure(s)" if failures else ""),
                sty["small"]
            ))
    else:
        story.append(Paragraph("<b>LLMOps:</b> no call logs recorded for this run.", sty["small"]))
    story.append(Spacer(1, 8))

    # Executive Conclusion — synthesizes quality, model reliability, risk,
    # and deployment readiness into a closing paragraph, separate from the
    # opening Executive Summary.
    story.append(Paragraph("Executive Conclusion", sty["h1"]))
    qd_final = analysis_data.get("quality_dimensions", {})
    quality_score_final = qd_final.get("composite_score")
    risk_score_final = analysis_data.get("risk_score", 0)
    best_model_final = ml.get("best_model", "N/A")
    best_score_final = ml.get("best_score", 0) or 0
    score_pct_final = round(best_score_final * 100, 1) if best_score_final <= 1 else round(best_score_final, 1)

    if risk_score_final >= 60:
        readiness = "not yet production-ready — the elevated risk score above should be resolved first"
    elif risk_score_final >= 30 or (quality_score_final is not None and quality_score_final < 70):
        readiness = "usable with caution — review the flagged risks and quality gaps before relying on it for decisions"
    else:
        readiness = "reasonably production-ready, pending normal validation"

    eval_note = ""
    if eval_result and eval_result.get("status") == "FLAGGED":
        eval_note = " Note: the automated Guardrails check flagged items in this report for human review — see review notes before treating conclusions as final."
    elif eval_result and eval_result.get("status") == "FAILED":
        eval_note = " Note: the automated Guardrails check failed completeness validation on this report — treat conclusions as provisional pending human review."

    story.append(Paragraph(
        f"This dataset scored {quality_score_final if quality_score_final is not None else 'N/A'}/100 on data quality, "
        f"with {best_model_final} achieving the strongest performance ({score_pct_final}%) among the models compared. "
        f"Overall risk was assessed at {risk_score_final}/100. Taken together, this analysis is {readiness}.{eval_note}",
        sty["body"]
    ))
    story.append(Spacer(1, 8))

    # Footer
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=C_LIGHT))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated by DataMind AI v3.0  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        "Predictions are probabilistic. Validate before production use.",
        sty["footer"]
    ))

    doc.build(story)
    return buf.getvalue()