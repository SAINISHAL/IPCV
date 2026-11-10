from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(r"D:\IpCv\Classroom-Eigenfaces-Recognition")
OUT_PATH = PROJECT_ROOT / "Eigenfaces_Presentation.pptx"
ASSET_DIR = PROJECT_ROOT / "ppt_assets"
ASSET_DIR.mkdir(exist_ok=True)

COLORS = {
    "navy": RGBColor(16, 34, 68),
    "blue": RGBColor(37, 99, 235),
    "cyan": RGBColor(56, 189, 248),
    "teal": RGBColor(13, 148, 136),
    "light": RGBColor(239, 246, 255),
    "gray": RGBColor(79, 91, 102),
    "white": RGBColor(255, 255, 255),
    "gold": RGBColor(245, 158, 11),
    "red": RGBColor(220, 38, 127),
    "green": RGBColor(22, 163, 74),
    "black": RGBColor(15, 23, 42),
}


def generate_face_image(path: Path, style: str = "normal"):
    w, h = 500, 500
    img = Image.new("L", (w, h), 240)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    face_w, face_h = 240, 300
    face_box = [cx - face_w // 2, cy - face_h // 2, cx + face_w // 2, cy + face_h // 2]
    draw.ellipse(face_box, fill=220, outline=80, width=4)

    draw.ellipse((face_box[0] - 28, cy - 70, face_box[0] + 10, cy + 70), fill=210)
    draw.ellipse((face_box[2] - 10, cy - 70, face_box[2] + 28, cy + 70), fill=210)

    draw.ellipse((cx - 140, cy - 200, cx + 140, cy + 30), fill=115)
    draw.arc((cx - 80, cy - 120, cx + 80, cy + 40), start=180, end=0, fill=80, width=10)

    eye_y = cy - 40
    eye_w = 42
    draw.ellipse((cx - 90 - eye_w, eye_y - 12, cx - 90 + eye_w, eye_y + 12), fill=70)
    draw.ellipse((cx + 90 - eye_w, eye_y - 12, cx + 90 + eye_w, eye_y + 12), fill=70)
    if style == "eigen":
        draw.ellipse((cx - 90 - eye_w, eye_y - 12, cx - 90 + eye_w, eye_y + 12), fill=110)
        draw.ellipse((cx + 90 - eye_w, eye_y - 12, cx + 90 + eye_w, eye_y + 12), fill=110)

    draw.line((cx, cy - 10, cx, cy + 70), fill=100, width=5)
    draw.polygon([(cx - 8, cy + 78), (cx + 8, cy + 78), (cx, cy + 110)], fill=120)
    draw.arc((cx - 55, cy + 85, cx + 55, cy + 135), start=200, end=340, fill=90, width=6)

    if style == "mean":
        alpha = Image.new("L", (w, h), 0)
        for y in range(h):
            for x in range(w):
                dist = math.hypot(x - cx, y - cy)
                if dist < 200:
                    val = int(255 - dist / 2.5)
                    alpha.putpixel((x, y), max(0, min(255, val)))
        img = Image.blend(img, alpha, 0.18)
    elif style == "eigen":
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        img = Image.eval(img, lambda p: p * 0.9)

    img.save(path)


def build_assets():
    generate_face_image(ASSET_DIR / "face_vector.png", "normal")
    generate_face_image(ASSET_DIR / "mean_face.png", "mean")
    generate_face_image(ASSET_DIR / "eigenface.png", "eigen")


def add_title_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = COLORS["light"]

    banner = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.8))
    banner.fill.solid(); banner.fill.fore_color.rgb = COLORS["navy"]
    banner.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.1), Inches(12.2), Inches(1.0))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Eigenfaces for Face Recognition"
    p.alignment = PP_ALIGN.LEFT
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = COLORS["navy"]

    subtitle = slide.shapes.add_textbox(Inches(0.7), Inches(2.0), Inches(11.8), Inches(0.8))
    tf2 = subtitle.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = "From image matrices to facial identity in a classroom recognition system"
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLORS["gray"]

    left = slide.shapes.add_shape(1, Inches(1.0), Inches(3.1), Inches(2.6), Inches(2.8))
    left.fill.solid(); left.fill.fore_color.rgb = COLORS["white"]
    left.line.color.rgb = COLORS["blue"]
    left.line.width = 2
    left.text = "Face image"

    center = slide.shapes.add_shape(1, Inches(4.0), Inches(3.1), Inches(2.6), Inches(2.8))
    center.fill.solid(); center.fill.fore_color.rgb = COLORS["light"]
    center.line.color.rgb = COLORS["teal"]
    center.line.width = 2
    center.text = "Face as vector"

    right = slide.shapes.add_shape(1, Inches(7.0), Inches(3.1), Inches(2.8), Inches(2.8))
    right.fill.solid(); right.fill.fore_color.rgb = COLORS["blue"]
    right.line.color.rgb = COLORS["navy"]
    right.line.width = 2
    right.text = "Eigenface basis"

    summary = slide.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(11.8), Inches(0.9))
    tf3 = summary.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "Topic: PCA-based facial subspace learning | Project use case: classroom attendance and identity recognition"
    p3.font.size = Pt(15)
    p3.font.color.rgb = COLORS["navy"]


def add_slide_why(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(8.0), Inches(0.5))
    t = title.text_frame.paragraphs[0]
    t.text = "Why face recognition is difficult"
    t.font.size = Pt(24)
    t.font.bold = True
    t.font.color.rgb = COLORS["navy"]

    panel = slide.shapes.add_shape(1, Inches(0.6), Inches(1.0), Inches(5.5), Inches(4.8))
    panel.fill.solid(); panel.fill.fore_color.rgb = COLORS["light"]
    panel.line.color.rgb = COLORS["blue"]
    panel.line.width = 1
    tf = panel.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "Same person can look different due to:"
    p1.font.bold = True; p1.font.size = Pt(20); p1.font.color.rgb = COLORS["navy"]
    for i, item in enumerate([
        "Pose changes",
        "Lighting variation",
        "Expression differences",
        "Occlusion / partial faces",
        "Camera angle and scale",
    ], start=1):
        p = tf.add_paragraph()
        p.text = f"{i}. {item}"
        p.font.size = Pt(18)
        p.font.color.rgb = COLORS["gray"]

    slide.shapes.add_picture(str(ASSET_DIR / "face_vector.png"), Inches(7.0), Inches(1.3), Inches(2.4), Inches(2.4))
    label = slide.shapes.add_textbox(Inches(7.0), Inches(3.8), Inches(2.4), Inches(0.4))
    label.text_frame.paragraphs[0].text = "Different images, same identity"
    label.text_frame.paragraphs[0].font.size = Pt(14)
    label.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    bar = slide.shapes.add_shape(1, Inches(0.7), Inches(5.9), Inches(11.7), Inches(1.1))
    bar.fill.solid(); bar.fill.fore_color.rgb = COLORS["navy"]
    bar.line.fill.background()
    tf2 = bar.text_frame
    p = tf2.paragraphs[0]
    p.text = "Goal: represent identity using a compact set of stable facial features instead of raw pixels."
    p.font.size = Pt(18); p.font.color.rgb = COLORS["white"]


def add_slide_face_as_vector(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(8.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "1. Face image as a vector"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    slide.shapes.add_picture(str(ASSET_DIR / "face_vector.png"), Inches(0.8), Inches(1.3), Inches(4.2), Inches(4.2))
    box = slide.shapes.add_shape(1, Inches(5.5), Inches(1.4), Inches(6.7), Inches(3.0))
    box.fill.solid(); box.fill.fore_color.rgb = COLORS["light"]
    box.line.color.rgb = COLORS["blue"]
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "A grayscale face is flattened into a single column vector x ∈ R^D"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLORS["navy"]
    p2 = tf.add_paragraph()
    p2.text = "For 100 × 100 face image, D = 10,000 pixels"
    p2.font.size = Pt(18)
    p2.font.color.rgb = COLORS["gray"]
    p3 = tf.add_paragraph()
    p3.text = "This creates a high-dimensional representation where every pixel contributes to identity and noise."
    p3.font.size = Pt(17)
    p3.font.color.rgb = COLORS["gray"]

    eq = slide.shapes.add_textbox(Inches(5.8), Inches(5.0), Inches(6.4), Inches(0.8))
    eq.text_frame.paragraphs[0].text = "x = [x1, x2, x3, ..., xD]^T"
    eq.text_frame.paragraphs[0].font.size = Pt(22)
    eq.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]


def add_slide_mean_face(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(8.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "2. Mean face and centered data"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    slide.shapes.add_picture(str(ASSET_DIR / "face_vector.png"), Inches(0.7), Inches(1.4), Inches(2.2), Inches(2.2))
    slide.shapes.add_picture(str(ASSET_DIR / "mean_face.png"), Inches(3.3), Inches(1.4), Inches(2.2), Inches(2.2))
    label1 = slide.shapes.add_textbox(Inches(0.7), Inches(3.8), Inches(2.1), Inches(0.5))
    label1.text_frame.paragraphs[0].text = "Individual face"
    label1.text_frame.paragraphs[0].font.size = Pt(16)
    label1.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]
    label2 = slide.shapes.add_textbox(Inches(3.3), Inches(3.8), Inches(2.1), Inches(0.5))
    label2.text_frame.paragraphs[0].text = "Average face"
    label2.text_frame.paragraphs[0].font.size = Pt(16)
    label2.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    box = slide.shapes.add_shape(1, Inches(6.1), Inches(1.4), Inches(5.9), Inches(3.7))
    box.fill.solid(); box.fill.fore_color.rgb = COLORS["light"]
    box.line.color.rgb = COLORS["blue"]
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Each face is centered by subtracting the mean face"
    p.font.size = Pt(19)
    p.font.bold = True
    p.font.color.rgb = COLORS["navy"]
    p2 = tf.add_paragraph()
    p2.text = "μ = (1/N) Σ xi"
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLORS["navy"]
    p3 = tf.add_paragraph()
    p3.text = "ai = xi − μ"
    p3.font.size = Pt(20)
    p3.font.color.rgb = COLORS["navy"]
    p4 = tf.add_paragraph()
    p4.text = "This removes common illumination and average face structure, leaving variation that carries identity."
    p4.font.size = Pt(16)
    p4.font.color.rgb = COLORS["gray"]


def add_slide_math(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(10.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "3. The Eigenfaces idea: principal directions of variation"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    left = slide.shapes.add_shape(1, Inches(0.7), Inches(1.1), Inches(5.2), Inches(5.1))
    left.fill.solid(); left.fill.fore_color.rgb = COLORS["light"]
    left.line.color.rgb = COLORS["blue"]
    tf = left.text_frame
    tf.word_wrap = True
    sequence = [
        "A = [a1, a2, ..., aN]",
        "L = A^T A",
        "L v_i = λ_i v_i",
        "u_i = A v_i",
        "u_i is the i-th eigenface",
        "The eigenfaces form a basis for the face subspace.",
    ]
    for i, line in enumerate(sequence):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(22)
        p.font.color.rgb = COLORS["navy"]
        p.font.bold = i in {2, 3, 4}

    right = slide.shapes.add_picture(str(ASSET_DIR / "eigenface.png"), Inches(7.2), Inches(1.8), Inches(2.9), Inches(2.9))
    cap = slide.shapes.add_textbox(Inches(7.0), Inches(4.9), Inches(3.5), Inches(0.6))
    cap.text_frame.paragraphs[0].text = "First eigenface: dominant pattern of variation"
    cap.text_frame.paragraphs[0].font.size = Pt(15)
    cap.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    note = slide.shapes.add_shape(1, Inches(0.8), Inches(6.5), Inches(11.4), Inches(0.9))
    note.fill.solid(); note.fill.fore_color.rgb = COLORS["navy"]
    note.line.fill.background()
    note.text = "Important idea: A face is not stored as all pixels. It is projected into a smaller, more meaningful eigenvector space."
    note.text_frame.paragraphs[0].font.size = Pt(17)
    note.text_frame.paragraphs[0].font.color.rgb = COLORS["white"]


def add_slide_project_pipeline(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(8.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "4. How Eigenfaces are used in this project"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    stages = [
        ("1. Detection", "YOLOv8-Face + YuNet detect all faces in classroom photos"),
        ("2. Alignment", "Normalize face pose using eye-level alignment and crop scaling"),
        ("3. Preprocess", "Resize to 100×100, grayscale, CLAHE, blur filtering"),
        ("4. PCA projection", "Project each face onto learned eigenface basis"),
        ("5. Classification", "Use SVM/KNN/Euclidean distance to recognize the student"),
        ("6. Decision", "Accept recognized name or reject as UNKNOWN"),
    ]

    x_start = 0.8
    y_start = 1.2
    w = 1.8
    h = 1.0
    for i, (name, desc) in enumerate(stages):
        box = slide.shapes.add_shape(1, Inches(x_start + (i % 3) * 4.2), Inches(y_start + (i // 3) * 1.9), Inches(w), Inches(h))
        box.fill.solid(); box.fill.fore_color.rgb = COLORS["light"] if i % 2 == 0 else COLORS["white"]
        box.line.color.rgb = COLORS["blue"]
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = f"{name}\n{desc}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLORS["navy"]

    consider = slide.shapes.add_shape(1, Inches(0.8), Inches(5.8), Inches(11.6), Inches(0.9))
    consider.fill.solid(); consider.fill.fore_color.rgb = COLORS["navy"]
    consider.line.fill.background()
    consider.text = "Main idea: the trained system learns the most important facial variation patterns and compares new faces against that compact subspace."
    consider.text_frame.paragraphs[0].font.size = Pt(17)
    consider.text_frame.paragraphs[0].font.color.rgb = COLORS["white"]


def add_slide_dataset_usage(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(8.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "5. Dataset and preprocessing in the classroom system"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    left = slide.shapes.add_shape(1, Inches(0.7), Inches(1.4), Inches(5.0), Inches(4.5))
    left.fill.solid(); left.fill.fore_color.rgb = COLORS["light"]
    left.line.color.rgb = COLORS["blue"]
    tf = left.text_frame
    p = tf.paragraphs[0]
    p.text = "Project workflow"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLORS["navy"]
    for item in [
        "Face crops are extracted from classroom images",
        "Students are clustered and mapped to identities",
        "Images are aligned and standardized",
        "PCA learns the eigenface basis from training samples",
        "Recognition is then done on unseen faces",
    ]:
        p2 = tf.add_paragraph(); p2.text = "• " + item; p2.font.size = Pt(17); p2.font.color.rgb = COLORS["gray"]

    right = slide.shapes.add_shape(1, Inches(6.2), Inches(1.4), Inches(5.8), Inches(4.0))
    right.fill.solid(); right.fill.fore_color.rgb = COLORS["white"]
    right.line.color.rgb = COLORS["teal"]
    tf2 = right.text_frame
    p = tf2.paragraphs[0]
    p.text = "Why preprocessing matters"
    p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLORS["navy"]
    for item in [
        "100 × 100 grayscale normalization",
        "CLAHE histogram equalization",
        "Background suppression and face padding",
        "Blur and low-quality filtering",
        "Eye alignment for better PCA consistency",
    ]:
        p2 = tf2.add_paragraph(); p2.text = "• " + item; p2.font.size = Pt(16); p2.font.color.rgb = COLORS["gray"]


def add_slide_classifier(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(10.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "6. Projection and classification"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    left = slide.shapes.add_shape(1, Inches(0.8), Inches(1.3), Inches(5.2), Inches(4.6))
    left.fill.solid(); left.fill.fore_color.rgb = COLORS["light"]
    left.line.color.rgb = COLORS["blue"]
    tf = left.text_frame
    p = tf.paragraphs[0]; p.text = "Projection step"; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLORS["navy"]
    p2 = tf.add_paragraph(); p2.text = "ω = U_K^T (x − μ)"; p2.font.size = Pt(22); p2.font.color.rgb = COLORS["navy"]
    p3 = tf.add_paragraph(); p3.text = "The face is represented by a compact weight vector in the principal component space."; p3.font.size = Pt(16); p3.font.color.rgb = COLORS["gray"]
    p4 = tf.add_paragraph(); p4.text = "This keeps only the most informative features and discards noise."; p4.font.size = Pt(16); p4.font.color.rgb = COLORS["gray"]

    right = slide.shapes.add_shape(1, Inches(6.4), Inches(1.3), Inches(5.8), Inches(4.6))
    right.fill.solid(); right.fill.fore_color.rgb = COLORS["white"]
    right.line.color.rgb = COLORS["navy"]
    tf2 = right.text_frame
    p = tf2.paragraphs[0]; p.text = "Recognition"; p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = COLORS["navy"]
    for item in [
        "Euclidean distance compares similarity to known face vectors",
        "KNN uses nearest neighbors in the PCA feature space",
        "SVM learns a decision boundary between student classes",
        "Unknown faces can be rejected when confidence is low",
    ]:
        p2 = tf2.add_paragraph(); p2.text = "• " + item; p2.font.size = Pt(16); p2.font.color.rgb = COLORS["gray"]

    note = slide.shapes.add_shape(1, Inches(1.0), Inches(6.3), Inches(11.2), Inches(0.8))
    note.fill.solid(); note.fill.fore_color.rgb = COLORS["navy"]
    note.line.fill.background()
    note.text = "In this project, the trained PCA model is paired with an SVM classifier and a confidence threshold to flag uncertain matches."
    note.text_frame.paragraphs[0].font.size = Pt(17)
    note.text_frame.paragraphs[0].font.color.rgb = COLORS["white"]


def add_slide_results(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(8.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "7. Performance and results"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    table = slide.shapes.add_shape(1, Inches(0.8), Inches(1.1), Inches(11.3), Inches(4.5))
    table.fill.solid(); table.fill.fore_color.rgb = COLORS["light"]
    table.line.color.rgb = COLORS["blue"]
    tf = table.text_frame
    tf.clear()
    rows = [
        ("Classifier", "Accuracy", "Macro F1", "Observation"),
        ("Euclidean", "68.25%", "0.6319", "Fast and simple similarity matching"),
        ("KNN", "66.67%", "0.5997", "Good local neighbors, more errors"),
        ("SVM", "69.84%", "0.7385", "Best balance among tested models"),
    ]
    for idx, row in enumerate(rows):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = " | ".join(row)
        p.font.size = Pt(17 if idx == 0 else 16)
        p.font.bold = idx == 0
        p.font.color.rgb = COLORS["navy"]

    note = slide.shapes.add_shape(1, Inches(0.9), Inches(5.9), Inches(11.2), Inches(0.9))
    note.fill.solid(); note.fill.fore_color.rgb = COLORS["navy"]
    note.line.fill.background()
    note.text = "The project used K = 80 principal components and paired Eigenfaces with a calibrated SVM to identify students while rejecting uncertain or unknown faces."
    note.text_frame.paragraphs[0].font.size = Pt(17)
    note.text_frame.paragraphs[0].font.color.rgb = COLORS["white"]


def add_slide_values(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(11.0), Inches(0.5))
    title.text_frame.paragraphs[0].text = "8. Why Eigenfaces still matters"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    cards = [
        ("Simple and explainable", "Internally interpretable and grounded in linear algebra"),
        ("Low-dimensional representation", "A face can be described by a small number of PCA weights"),
        ("Good for constrained data", "Strong for classroom or enrolled-identity recognition"),
        ("Fast and light", "No heavy deep-net embedding required at inference"),
    ]

    for i, (h, d) in enumerate(cards):
        box = slide.shapes.add_shape(1, Inches(0.8 + (i % 2) * 5.9), Inches(1.2 + (i // 2) * 2.3), Inches(4.5), Inches(1.4))
        box.fill.solid(); box.fill.fore_color.rgb = COLORS["light"]
        box.line.color.rgb = COLORS["blue"]
        tf = box.text_frame
        p = tf.paragraphs[0]; p.text = h; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = COLORS["navy"]
        p2 = tf.add_paragraph(); p2.text = d; p2.font.size = Pt(14); p2.font.color.rgb = COLORS["gray"]

    final = slide.shapes.add_shape(1, Inches(0.9), Inches(6.2), Inches(11.3), Inches(1.0))
    final.fill.solid(); final.fill.fore_color.rgb = COLORS["navy"]
    final.line.fill.background()
    final.text = "Even today, Eigenfaces remain a clear educational example of dimensionality reduction, subspace learning, and face modeling."
    final.text_frame.paragraphs[0].font.size = Pt(19)
    final.text_frame.paragraphs[0].font.color.rgb = COLORS["white"]


def add_slide_conclusion(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = COLORS["white"]

    title = slide.shapes.add_textbox(Inches(0.6), Inches(0.5), Inches(9.0), Inches(0.6))
    title.text_frame.paragraphs[0].text = "Conclusion"
    title.text_frame.paragraphs[0].font.size = Pt(28)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]

    box = slide.shapes.add_shape(1, Inches(1.0), Inches(1.4), Inches(11.0), Inches(4.2))
    box.fill.solid(); box.fill.fore_color.rgb = COLORS["light"]
    box.line.color.rgb = COLORS["blue"]
    tf = box.text_frame
    items = [
        "Eigenfaces convert raw face images into a compact mathematical subspace.",
        "The method learns the dominant directions of facial variation instead of memorizing pixels.",
        "In classroom recognition, this helps detect, normalize, compare, and classify faces efficiently.",
        "The project combines PCA with detection, preprocessing, and SVM-based decision-making to support attendance recognition.",
    ]
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(20)
        p.font.color.rgb = COLORS["gray"]

    end = slide.shapes.add_textbox(Inches(1.2), Inches(6.1), Inches(8.5), Inches(0.7))
    end.text_frame.paragraphs[0].text = "Thank you!"
    end.text_frame.paragraphs[0].font.size = Pt(26)
    end.text_frame.paragraphs[0].font.bold = True
    end.text_frame.paragraphs[0].font.color.rgb = COLORS["navy"]


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    build_assets()
    add_title_slide(prs)
    add_slide_why(prs)
    add_slide_face_as_vector(prs)
    add_slide_mean_face(prs)
    add_slide_math(prs)
    add_slide_project_pipeline(prs)
    add_slide_dataset_usage(prs)
    add_slide_classifier(prs)
    add_slide_results(prs)
    add_slide_values(prs)
    add_slide_conclusion(prs)

    prs.save(OUT_PATH)
    print(f"Created presentation: {OUT_PATH}")


if __name__ == "__main__":
    build_presentation()
