import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def safe_text(value):
    """
    Converts None/empty values into clean display text.
    """
    if value is None:
        return ""
    return str(value).strip()


def paragraph_list(items, style):
    """
    Converts a list of strings into numbered ReportLab paragraphs.
    """
    story = []

    if not items:
        story.append(Paragraph("No major items identified.", style))
        return story

    for i, item in enumerate(items, start=1):
        story.append(Paragraph(f"{i}. {safe_text(item)}", style))
        story.append(Spacer(1, 4))

    return story


def generate_report_pdf_bytes(patient_data: dict, result_data: dict) -> bytes:
    """
    Generates a downloadable prior authorization analysis report as PDF bytes.

    This function does not save anything to disk.
    It creates the PDF in memory so Streamlit can download it directly.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        alignment=1,
        textColor=colors.HexColor("#1a3350"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#666666"),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1a3350"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=3,
    )

    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#888888"),
    )

    story = []

    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    story.append(Paragraph("PreAuth.ai Prior Authorization Analysis Report", title_style))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            subtitle_style,
        )
    )

    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#1a3350")))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Risk summary table
    # -------------------------------------------------------------------------

    risk_score = result_data.get("risk_score", 0)
    risk_level = safe_text(result_data.get("risk_level", "unknown")).upper()

    risk_color = {
        "LOW": "#1a7f37",
        "MEDIUM": "#b26a00",
        "HIGH": "#c0392b",
    }.get(risk_level, "#1a3350")

    summary_table_data = [
        ["Risk Score", "Risk Level", "Patient", "Member ID"],
        [
            str(risk_score),
            risk_level,
            safe_text(patient_data.get("patient_name", "")),
            safe_text(patient_data.get("insurance_id", "")),
        ],
    ]

    summary_table = Table(
        summary_table_data,
        colWidths=[1.2 * inch, 1.4 * inch, 2.3 * inch, 1.8 * inch],
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3350")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 1), (0, 1), colors.HexColor(risk_color)),
                ("TEXTCOLOR", (0, 1), (0, 1), colors.white),
                ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (0, 1), 16),
                ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
                ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(risk_color)),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # Patient details
    # -------------------------------------------------------------------------

    story.append(Paragraph("Patient and Request Details", section_style))

    details = [
        ("Patient Name", patient_data.get("patient_name", "")),
        ("Insurance Member ID", patient_data.get("insurance_id", "")),
        ("Provider NPI", patient_data.get("provider_npi", "")),
        ("Requested Service", patient_data.get("drug_name", "")),
        ("Diagnosis Code", patient_data.get("diagnosis_code", "")),
    ]

    for label, value in details:
        story.append(Paragraph(f"<b>{label}:</b> {safe_text(value)}", body_style))

    if patient_data.get("clinical_summary"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Clinical Summary:</b>", body_style))
        story.append(Paragraph(safe_text(patient_data.get("clinical_summary")), body_style))

    # -------------------------------------------------------------------------
    # Executive summary
    # -------------------------------------------------------------------------

    story.append(Paragraph("Executive Summary", section_style))
    story.append(Paragraph(safe_text(result_data.get("summary", "")), body_style))

    # -------------------------------------------------------------------------
    # Denial reasons
    # -------------------------------------------------------------------------

    story.append(Paragraph("Denial Reasons", section_style))
    story.extend(paragraph_list(result_data.get("denial_reasons", []), body_style))

    # -------------------------------------------------------------------------
    # Missing documentation
    # -------------------------------------------------------------------------

    story.append(Paragraph("Missing Documentation", section_style))
    story.extend(paragraph_list(result_data.get("missing_documentation", []), body_style))

    # -------------------------------------------------------------------------
    # Recommended fixes
    # -------------------------------------------------------------------------

    story.append(Paragraph("Recommended Fixes Before Submission", section_style))
    story.extend(paragraph_list(result_data.get("recommended_fixes", []), body_style))

    # -------------------------------------------------------------------------
    # Retrieved policy sources
    # -------------------------------------------------------------------------

    story.append(Paragraph("Retrieved Policy Sources", section_style))

    sources = result_data.get("retrieved_sources", [])

    if not sources:
        story.append(Paragraph("No retrieved sources available.", small_style))
    else:
        for i, source in enumerate(sources, start=1):
            source_text = (
                f"{i}. Source: {safe_text(source.get('source_file'))} | "
                f"Folder: {safe_text(source.get('source_folder'))} | "
                f"Chunk: {safe_text(source.get('chunk_index'))} | "
                f"Type: {safe_text(source.get('query_label'))}"
            )
            story.append(Paragraph(source_text, small_style))

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Generated by PreAuth.ai. For educational/demo use only. Provider review is required before payer submission.",
            footer_style,
        )
    )

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()