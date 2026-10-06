"""
generate_pdf_report.py
Generates a polished, multi-page executive PDF report:
reports/data_cleaning_report.pdf
Utilizing ReportLab with custom styling, structured data tables, and embedded high-res figures.
"""

import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Data Cleaning Project — Enterprise Customer Analytics Pipeline")
            self.drawRightString(558, 755, "Week 2 Task 1 Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential — Data Science & Machine Learning Engineering Track")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf_report(base_dir: str):
    pdf_path = os.path.join(base_dir, "reports", "data_cleaning_report.pdf")
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Enterprise Data Cleaning & Quality Assurance Report", title_style))
    story.append(Paragraph("<b>Track</b>: Data Science Internship — Week 2 Task 1 &nbsp;|&nbsp; <b>Author</b>: Data Science Intern &nbsp;|&nbsp; <b>Date</b>: October 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    exec_summary = (
        "In modern predictive analytics and machine learning, real-world data is inherently noisy, incomplete, "
        "and fragmented. This report details the comprehensive data cleaning lifecycle implemented on a raw customer churn "
        "dataset containing <b>1,550 raw records</b> across 16 commercial and demographic attributes. Through a structured "
        "10-step protocol, we audited data defects, eliminated duplicates, standardized headers, imputed 1,522 missing entries, "
        "neutralized extreme outliers via IQR Winsorization, and produced a verified analysis-ready dataset of <b>1,413 pristine records</b>."
    )
    story.append(Paragraph(exec_summary, body_style))

    # KPI summary cards in a table
    kpi_data = [
        [
            Paragraph("<b>Raw Ingested Records</b><br/><font size=12 color='#1E3A8A'><b>1,550</b></font>", body_style),
            Paragraph("<b>Duplicates Purged</b><br/><font size=12 color='#DC2626'><b>35 exact + 14 PK</b></font>", body_style),
            Paragraph("<b>Missing Values Resolved</b><br/><font size=12 color='#059669'><b>1,522 cells (100%)</b></font>", body_style),
            Paragraph("<b>Final Clean Records</b><br/><font size=12 color='#2563EB'><b>1,413 (100% Valid)</b></font>", body_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[126, 126, 126, 126])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # 2. Missing Data Diagnostics
    story.append(Paragraph("2. Missing Data Diagnostics & Heatmap Analysis", h1_style))
    story.append(Paragraph(
        "A missing data heatmap was constructed to evaluate null distributions. 14 out of 16 columns exhibited missingness "
        "rates ranging from 1.6% to 14.3%. Missingness was addressed through statistical medians for continuous attributes, "
        "domain-based formula reconstruction for charges, and explicit categorical classifications.", body_style
    ))

    heatmap_img_path = os.path.join(base_dir, "outputs", "figures", "01_missing_data_heatmap.png")
    if os.path.exists(heatmap_img_path):
        story.append(Image(heatmap_img_path, width=490, height=195))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 3. Strategy Matrix Table
    story.append(Paragraph("3. Systematic Column-by-Column Imputation Strategy", h1_style))
    strat_data = [
        [
            Paragraph("<b>Feature</b>", table_header_style),
            Paragraph("<b>Null Count</b>", table_header_style),
            Paragraph("<b>Strategy Applied</b>", table_header_style),
            Paragraph("<b>Decision Rationale</b>", table_header_style)
        ],
        [
            Paragraph("<b>customer_id</b>", table_cell_bold),
            Paragraph("26 (1.7%)", table_cell_style),
            Paragraph("Drop Rows", table_cell_style),
            Paragraph("Primary key cannot be fabricated; prevents transaction cross-contamination.", table_cell_style)
        ],
        [
            Paragraph("<b>churn</b>", table_cell_bold),
            Paragraph("62 (4.4%)", table_cell_style),
            Paragraph("Drop Rows", table_cell_style),
            Paragraph("Predictive modeling target; imputing ground truth introduces synthetic bias.", table_cell_style)
        ],
        [
            Paragraph("<b>annual_income</b>", table_cell_bold),
            Paragraph("120 (7.7%)", table_cell_style),
            Paragraph("Median ($87,396)", table_cell_style),
            Paragraph("Heavily skewed distribution; median avoids high-income distortion.", table_cell_style)
        ],
        [
            Paragraph("<b>age</b>", table_cell_bold),
            Paragraph("101 (6.5%)", table_cell_style),
            Paragraph("Median (41 yrs)", table_cell_style),
            Paragraph("Cast to int64; corrected negative entry errors and typos >120.", table_cell_style)
        ],
        [
            Paragraph("<b>tenure_months</b>", table_cell_bold),
            Paragraph("109 (7.0%)", table_cell_style),
            Paragraph("Median (37 mos)", table_cell_style),
            Paragraph("Customer lifecycle metric; clipped negative values to 0.", table_cell_style)
        ],
        [
            Paragraph("<b>monthly_charges</b>", table_cell_bold),
            Paragraph("92 (5.9%)", table_cell_style),
            Paragraph("Median ($70.97)", table_cell_style),
            Paragraph("Replaced zero/negative entries; imputed median.", table_cell_style)
        ],
        [
            Paragraph("<b>total_charges</b>", table_cell_bold),
            Paragraph("174 (11.2%)", table_cell_style),
            Paragraph("tenure * monthly", table_cell_style),
            Paragraph("Reconstructed via mathematical domain formula rather than arbitrary guess.", table_cell_style)
        ],
        [
            Paragraph("<b>join_date</b>", table_cell_bold),
            Paragraph("120 (7.7%)", table_cell_style),
            Paragraph("Tenure Backfill", table_cell_style),
            Paragraph("Calculated as reference date (2024-12-31) minus tenure days.", table_cell_style)
        ],
        [
            Paragraph("<b>payment_method</b>", table_cell_bold),
            Paragraph("91 (5.9%)", table_cell_style),
            Paragraph("'Unknown'", table_cell_style),
            Paragraph("Preserves transaction integrity without assuming user banking choices.", table_cell_style)
        ],
        [
            Paragraph("<b>gender</b>", table_cell_bold),
            Paragraph("83 (5.4%)", table_cell_style),
            Paragraph("'Unspecified'", table_cell_style),
            Paragraph("Maintains demographic audit trail; complies with privacy practices.", table_cell_style)
        ]
    ]
    strat_table = Table(strat_data, colWidths=[85, 65, 95, 255])
    strat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(strat_table)
    story.append(Spacer(1, 10))

    # 4. Outlier Analysis & Treatment
    story.append(Paragraph("4. Outlier Detection & IQR Winsorization Treatment", h1_style))
    story.append(Paragraph(
        "Outliers were evaluated via both Interquartile Range (IQR) and Z-Score (|Z| > 3). Severe billing anomalies "
        "($999.99/mo) and multi-million incomes were capped using Winsorization [Q1 - 1.5*IQR, Q3 + 1.5*IQR] to stabilize "
        "variance while avoiding deleting valuable accounts.", body_style
    ))

    outlier_before_img = os.path.join(base_dir, "outputs", "figures", "03_outliers_before_boxplots.png")
    outlier_after_img = os.path.join(base_dir, "outputs", "figures", "04_outliers_after_boxplots.png")
    if os.path.exists(outlier_before_img) and os.path.exists(outlier_after_img):
        story.append(Image(outlier_before_img, width=490, height=135))
        story.append(Spacer(1, 4))
        story.append(Image(outlier_after_img, width=490, height=135))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # 5. Post-Cleaning Distributions & Visualizations
    story.append(Paragraph("5. Post-Cleaning Feature Distributions", h1_style))
    story.append(Paragraph(
        "Following imputation, type conversions, and IQR capping, distributions for all continuous numerical "
        "and ordinal attributes display smooth, realistic characteristics suitable for machine learning algorithms.", body_style
    ))

    dist_img_path = os.path.join(base_dir, "outputs", "figures", "05_numerical_distributions.png")
    if os.path.exists(dist_img_path):
        story.append(Image(dist_img_path, width=490, height=230))
        story.append(Spacer(1, 8))

    # 6. Before-and-After Comparison Table
    story.append(Paragraph("6. Before-and-After Data Quality Comparison Table", h1_style))
    
    comp_csv_path = os.path.join(base_dir, "outputs", "tables", "before_after_metrics.csv")
    if os.path.exists(comp_csv_path):
        cdf = pd.read_csv(comp_csv_path)
        comp_data = [[
            Paragraph("<b>Data Quality Dimension</b>", table_header_style),
            Paragraph("<b>Raw Dataset (Before)</b>", table_header_style),
            Paragraph("<b>Clean Dataset (After)</b>", table_header_style),
            Paragraph("<b>Delta / Transformation Impact</b>", table_header_style)
        ]]
        for _, row in cdf.iterrows():
            comp_data.append([
                Paragraph(f"<b>{row['Metric']}</b>", table_cell_bold),
                Paragraph(str(row['Before Cleaning (Raw)']), table_cell_style),
                Paragraph(str(row['After Cleaning']), table_cell_style),
                Paragraph(str(row['Delta / Impact']), table_cell_style)
            ])
        comp_table = Table(comp_data, colWidths=[120, 115, 105, 160])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('TOPPADDING', (0,0), (-1,-1), 3.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 8))

    # 7. Summary & Recommendations
    story.append(Paragraph("7. Quality Sign-Off & Production Recommendations", h1_style))
    recs = (
        "<b>1. Zero Defect Verification</b>: All 6 automated data integrity assertions passed (0 nulls, 0 duplicate keys, "
        "valid domains, positive financial values).<br/>"
        "<b>2. Encoding Pipeline</b>: Ordinal mapping applied to contract types (0, 1, 2) and binary mapping to churn (0, 1). "
        "One-hot encoding applied to multi-class nominal features with drop_first=True to avoid collinearity.<br/>"
        "<b>3. Persistent Deliverables</b>: Cleaned datasets available at <code>data/cleaned/customer_churn_cleaned.csv</code> "
        "and ML-ready encoded dataset at <code>data/cleaned/customer_churn_encoded.csv</code>."
    )
    story.append(Paragraph(recs, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated executive PDF report: {pdf_path}")

if __name__ == "__main__":
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_pdf_report(base_directory)
