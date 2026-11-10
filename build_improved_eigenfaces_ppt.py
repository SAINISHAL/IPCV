from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(r"D:\IpCv\Classroom-Eigenfaces-Recognition")
OUT_PATH = PROJECT_ROOT / "Eigenfaces_Improved_Presentation.pptx"
ASSET_DIR = PROJECT_ROOT / "ppt_assets_2"
ASSET_DIR.mkdir(exist_ok=True)

NAVY = RGBColor(12, 29, 52)
BLUE = RGBColor(46, 94, 205)
CYAN = RGBColor(82, 196, 255)
TEAL = RGBColor(16, 185, 129)
LIGHT = RGBColor(240, 246, 255)
GRAY = RGBColor(71, 85, 105)
WHITE = RGBColor(255, 255, 255)
GOLD = RGBColor(245, 158, 11)
RED = RGBColor(220, 38, 127)
GREEN = RGBColor(22, 163, 74)


def gen_face(path: Path, mode: str = "normal"):
    w, h = 500, 500
    img = Image.new("L", (w, h), 245)
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    face_box = (cx - 120, cy - 150, cx + 120, cy + 150)
    draw.ellipse(face_box, fill=220, outline=90, width=5)

    draw.ellipse((face_box[0] - 35, cy - 90, face_box[0] + 10, cy + 90), fill=210)
    draw.ellipse((face_box[2] - 10, cy - 90, face_box[2] + 35, cy + 90), fill=210)

    draw.ellipse((cx - 150, cy - 200, cx + 150, cy + 10), fill=120)
    draw.arc((cx - 80, cy - 120, cx + 80, cy + 20), start=180, end=0, fill=80, width=10)

    eye_y = cy - 40
    eye_w = 38
    draw.ellipse((cx - 90 - eye_w, eye_y - 12, cx - 90 + eye_w, eye_y + 12), fill=70)
    draw.ellipse((cx + 90 - eye_w, eye_y - 12, cx + 90 + eye_w, eye_y + 12), fill=70)
    draw.line((cx, cy - 10, cx, cy + 70), fill=120, width=5)
    draw.polygon([(cx - 5, cy + 80), (cx + 5, cy + 80), (cx, cy + 110)], fill=120)
    draw.arc((cx - 55, cy + 80, cx + 55, cy + 135), start=200, end=340, fill=95, width=6)

    if mode == "mean":
        alpha = Image.new("L", (w, h), 0)
        for y in range(h):
            for x in range(w):
                d = math.hypot(x - cx, y - cy)
                if d < 220:
                    alpha.putpixel((x, y), int(255 - d / 2.5))
        img = Image.blend(img, alpha, 0.18)
    elif mode == "eigen":
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.GaussianBlur(0.7))
        img = Image.eval(img, lambda p: int(p * 0.88))

    img.save(path)


def make_assets():
    gen_face(ASSET_DIR / "face1.png", "normal")
    gen_face(ASSET_DIR / "mean_face.png", "mean")
    gen_face(ASSET_DIR / "eigenface.png", "eigen")


def add_bg_title(slide, title_text, subtitle_text=None):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    top = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.7))
    top.fill.solid(); top.fill.fore_color.rgb = NAVY
    top.line.fill.background()

    title = slide.shapes.add_textbox(Inches(0.55), Inches(0.9), Inches(12.0), Inches(0.8))
    p = title.text_frame.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle_text:
        sub = slide.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(12.0), Inches(0.6))
        p2 = sub.text_frame.paragraphs[0]
        p2.text = subtitle_text
        p2.font.size = Pt(16)
        p2.font.color.rgb = GRAY


def add_slide_1(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "Eigenfaces for Face Recognition", "A classroom attendance system using PCA and face subspace learning")

    panel = slide.shapes.add_shape(1, Inches(0.85), Inches(2.6), Inches(9.7), Inches(2.7))
    panel.fill.solid(); panel.fill.fore_color.rgb = LIGHT
    panel.line.color.rgb = BLUE
    panel.line.width = 1
    tf = panel.text_frame
    tf.word_wrap = True
    t = tf.paragraphs[0]
    t.text = "Project goal"
    t.font.size = Pt(20)
    t.font.bold = True
    t.font.color.rgb = NAVY
    for item in [
        "Detect faces from classroom photos and webcam frames",
        "Normalize facial images for consistency",
        "Project faces into a lower-dimensional feature space",
        "Recognize enrolled students and reject unknowns",
    ]:
        p = tf.add_paragraph(); p.text = "• " + item; p.font.size = Pt(18); p.font.color.rgb = GRAY

    face = slide.shapes.add_picture(str(ASSET_DIR / "face1.png"), Inches(10.7), Inches(2.8), Inches(1.9), Inches(2.5))
    note = slide.shapes.add_textbox(Inches(0.9), Inches(5.8), Inches(11.0), Inches(0.7))
    note.text_frame.paragraphs[0].text = "The system combines face detection, preprocessing, PCA projection, and classification."
    note.text_frame.paragraphs[0].font.size = Pt(19)
    note.text_frame.paragraphs[0].font.color.rgb = NAVY


def add_slide_2(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "Why face recognition is difficult", "Same person looks different under different conditions")

    left = slide.shapes.add_shape(1, Inches(0.7), Inches(2.0), Inches(5.7), Inches(3.7))
    left.fill.solid(); left.fill.fore_color.rgb = LIGHT
    left.line.color.rgb = BLUE
    tf = left.text_frame
    tf.word_wrap = True
    heading = tf.paragraphs[0]; heading.text = "Challenges"; heading.font.size = Pt(22); heading.font.bold = True; heading.font.color.rgb = NAVY
    for i, item in enumerate([
        "Pose variation",
        "Lighting changes",
        "Expression differences",
        "Camera angle and scale",
        "Background clutter and occlusion",
    ], start=1):
        p = tf.add_paragraph(); p.text = f"{i}. {item}"; p.font.size = Pt(18); p.font.color.rgb = GRAY

    right = slide.shapes.add_shape(1, Inches(7.2), Inches(2.2), Inches(5.2), Inches(3.1))
    right.fill.solid(); right.fill.fore_color.rgb = WHITE
    right.line.color.rgb = TEAL
    tf2 = right.text_frame
    tf2.word_wrap = True
    p1 = tf2.paragraphs[0]; p1.text = "Problem:"; p1.font.size = Pt(22); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf2.add_paragraph(); p2.text = "The same person gives different pixel patterns in different conditions, so raw image comparison is unreliable."; p2.font.size = Pt(18); p2.font.color.rgb = GRAY

    bottom = slide.shapes.add_shape(1, Inches(0.8), Inches(6.0), Inches(11.6), Inches(0.8))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "Solution: Learn the dominant patterns of variation and compare faces in a compact subspace."
    bottom.text_frame.paragraphs[0].font.size = Pt(18)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_3(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "1. Face image as a vector", "A grayscale face is just a very long number vector")

    img = slide.shapes.add_picture(str(ASSET_DIR / "face1.png"), Inches(0.8), Inches(2.0), Inches(3.8), Inches(3.8))
    box = slide.shapes.add_shape(1, Inches(5.0), Inches(2.1), Inches(7.2), Inches(3.4))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = BLUE
    tf = box.text_frame
    t = tf.paragraphs[0]
    t.text = "Image → vector"
    t.font.size = Pt(22); t.font.bold = True; t.font.color.rgb = NAVY
    p1 = tf.add_paragraph(); p1.text = "If the image is 100 × 100 pixels, then"; p1.font.size = Pt(19); p1.font.color.rgb = GRAY
    p2 = tf.add_paragraph(); p2.text = "D = 100 × 100 = 10,000"; p2.font.size = Pt(22); p2.font.bold = True; p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph(); p3.text = "x = [x1, x2, x3, ..., xD]^T"; p3.font.size = Pt(20); p3.font.color.rgb = NAVY
    p4 = tf.add_paragraph(); p4.text = "Each pixel becomes one coordinate in a high-dimensional space."; p4.font.size = Pt(17); p4.font.color.rgb = GRAY

    note = slide.shapes.add_textbox(Inches(0.9), Inches(6.2), Inches(11.0), Inches(0.5))
    note.text_frame.paragraphs[0].text = "This is useful mathematically, but the space is very large and contains a lot of redundant information."
    note.text_frame.paragraphs[0].font.size = Pt(18)
    note.text_frame.paragraphs[0].font.color.rgb = NAVY


def add_slide_4(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "2. Mean face and centered data", "Remove the common pattern to reveal identity differences")

    left = slide.shapes.add_picture(str(ASSET_DIR / "face1.png"), Inches(0.8), Inches(2.0), Inches(2.4), Inches(2.4))
    mid = slide.shapes.add_picture(str(ASSET_DIR / "mean_face.png"), Inches(3.8), Inches(2.0), Inches(2.4), Inches(2.4))
    right = slide.shapes.add_shape(1, Inches(7.0), Inches(1.8), Inches(5.4), Inches(3.8))
    right.fill.solid(); right.fill.fore_color.rgb = LIGHT
    right.line.color.rgb = BLUE
    tf = right.text_frame
    p1 = tf.paragraphs[0]; p1.text = "Mean face"; p1.font.size = Pt(22); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf.add_paragraph(); p2.text = "μ = (1/N) Σ xi"; p2.font.size = Pt(22); p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph(); p3.text = "Each image is centered:"; p3.font.size = Pt(18); p3.font.color.rgb = GRAY
    p4 = tf.add_paragraph(); p4.text = "ai = xi − μ"; p4.font.size = Pt(22); p4.font.color.rgb = NAVY
    p5 = tf.add_paragraph(); p5.text = "This removes shared face structure and keeps variation that carries identity."; p5.font.size = Pt(16); p5.font.color.rgb = GRAY

    label1 = slide.shapes.add_textbox(Inches(0.8), Inches(4.8), Inches(2.4), Inches(0.4))
    label1.text_frame.paragraphs[0].text = "One face"
    label1.text_frame.paragraphs[0].font.size = Pt(16); label1.text_frame.paragraphs[0].font.color.rgb = NAVY
    label2 = slide.shapes.add_textbox(Inches(3.8), Inches(4.8), Inches(2.4), Inches(0.4))
    label2.text_frame.paragraphs[0].text = "Mean face"
    label2.text_frame.paragraphs[0].font.size = Pt(16); label2.text_frame.paragraphs[0].font.color.rgb = NAVY


def add_slide_5(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "3. Eigenfaces: principal directions of variation", "Learn the most important patterns in the dataset")

    left = slide.shapes.add_shape(1, Inches(0.7), Inches(1.9), Inches(5.5), Inches(4.4))
    left.fill.solid(); left.fill.fore_color.rgb = LIGHT
    left.line.color.rgb = BLUE
    tf = left.text_frame
    tf.word_wrap = True
    h = tf.paragraphs[0]; h.text = "Matrix formulation"; h.font.size = Pt(22); h.font.bold = True; h.font.color.rgb = NAVY
    for line in [
        "A = [a1, a2, ..., aN]",
        "L = A^T A",
        "L v_i = λ_i v_i",
        "u_i = A v_i",
        "u_i is the i-th eigenface",
    ]:
        p = tf.add_paragraph(); p.text = line; p.font.size = Pt(22); p.font.color.rgb = NAVY

    right = slide.shapes.add_picture(str(ASSET_DIR / "eigenface.png"), Inches(7.2), Inches(2.2), Inches(2.8), Inches(2.8))
    cap = slide.shapes.add_textbox(Inches(7.0), Inches(5.4), Inches(3.4), Inches(0.5))
    cap.text_frame.paragraphs[0].text = "Eigenface = dominating variation pattern"
    cap.text_frame.paragraphs[0].font.size = Pt(16)
    cap.text_frame.paragraphs[0].font.color.rgb = NAVY

    bottom = slide.shapes.add_shape(1, Inches(0.8), Inches(6.0), Inches(11.6), Inches(0.8))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "The first eigenfaces encode the strongest variations; later ones encode finer details such as eye position, nose shape, and expression."
    bottom.text_frame.paragraphs[0].font.size = Pt(17)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_6(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "4. Projection into face space", "Convert each face into compact PCA coefficients")

    left = slide.shapes.add_shape(1, Inches(0.7), Inches(2.0), Inches(5.5), Inches(3.7))
    left.fill.solid(); left.fill.fore_color.rgb = LIGHT
    left.line.color.rgb = BLUE
    tf = left.text_frame
    p1 = tf.paragraphs[0]; p1.text = "Projection"; p1.font.size = Pt(22); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf.add_paragraph(); p2.text = "ω = U_K^T (x − μ)"; p2.font.size = Pt(22); p2.font.color.rgb = NAVY
    p3 = tf.add_paragraph(); p3.text = "A face is represented by a small weight vector ω."; p3.font.size = Pt(18); p3.font.color.rgb = GRAY
    p4 = tf.add_paragraph(); p4.text = "Only the most informative facial features remain."; p4.font.size = Pt(18); p4.font.color.rgb = GRAY

    right = slide.shapes.add_shape(1, Inches(7.0), Inches(2.0), Inches(5.3), Inches(3.7))
    right.fill.solid(); right.fill.fore_color.rgb = WHITE
    right.line.color.rgb = TEAL
    tf2 = right.text_frame
    p = tf2.paragraphs[0]; p.text = "Why this helps"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = NAVY
    for item in [
        "Reduces dimensionality massively",
        "Eliminates noisy pixel variation",
        "Finds the principal facial patterns",
        "Comparisons become more stable and efficient",
    ]:
        p2 = tf2.add_paragraph(); p2.text = "• " + item; p2.font.size = Pt(17); p2.font.color.rgb = GRAY

    bottom = slide.shapes.add_shape(1, Inches(0.8), Inches(6.0), Inches(11.6), Inches(0.8))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "This is the key idea: instead of comparing raw image pixels, compare the low-dimensional PCA weights of each face."
    bottom.text_frame.paragraphs[0].font.size = Pt(17)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_7(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "5. Recognition in the Eigenface space", "Use the projected weight vector to classify new faces")

    top = slide.shapes.add_shape(1, Inches(0.8), Inches(2.1), Inches(11.7), Inches(1.2))
    top.fill.solid(); top.fill.fore_color.rgb = LIGHT
    top.line.color.rgb = BLUE
    tf = top.text_frame
    p = tf.paragraphs[0]; p.text = "A new face is projected into the same subspace and compared with trained identities."; p.font.size = Pt(20); p.font.color.rgb = NAVY

    box1 = slide.shapes.add_shape(1, Inches(0.9), Inches(3.8), Inches(3.2), Inches(1.8))
    box1.fill.solid(); box1.fill.fore_color.rgb = WHITE; box1.line.color.rgb = BLUE
    tf1 = box1.text_frame; p1 = tf1.paragraphs[0]; p1.text = "Euclidean distance"; p1.font.size = Pt(20); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf1.add_paragraph(); p2.text = "Closest matching face vector"; p2.font.size = Pt(16); p2.font.color.rgb = GRAY

    box2 = slide.shapes.add_shape(1, Inches(4.7), Inches(3.8), Inches(3.2), Inches(1.8))
    box2.fill.solid(); box2.fill.fore_color.rgb = WHITE; box2.line.color.rgb = BLUE
    tf2 = box2.text_frame; p1 = tf2.paragraphs[0]; p1.text = "KNN"; p1.font.size = Pt(20); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf2.add_paragraph(); p2.text = "Nearest neighbors in feature space"; p2.font.size = Pt(16); p2.font.color.rgb = GRAY

    box3 = slide.shapes.add_shape(1, Inches(8.4), Inches(3.8), Inches(3.8), Inches(1.8))
    box3.fill.solid(); box3.fill.fore_color.rgb = WHITE; box3.line.color.rgb = BLUE
    tf3 = box3.text_frame; p1 = tf3.paragraphs[0]; p1.text = "SVM"; p1.font.size = Pt(20); p1.font.bold = True; p1.font.color.rgb = NAVY
    p2 = tf3.add_paragraph(); p2.text = "Learns a decision boundary between student classes and rejects low-confidence predictions"; p2.font.size = Pt(16); p2.font.color.rgb = GRAY

    bottom = slide.shapes.add_shape(1, Inches(0.8), Inches(6.1), Inches(11.6), Inches(0.7))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "The model can return the student name or UNKNOWN when the confidence is too low."
    bottom.text_frame.paragraphs[0].font.size = Pt(18)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_8(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "6. How Eigenfaces is used in this project", "Real classroom recognition pipeline")

    steps = [
        ("1. Detection", "YOLOv8-Face + YuNet locate faces in group images"),
        ("2. Alignment", "Eye-level alignment and cropping improve consistency"),
        ("3. Preprocessing", "Resize to 100×100, grayscale, CLAHE, quality filtering"),
        ("4. PCA training", "Build mean face and eigenfaces from known student images"),
        ("5. Projection", "Project each face into the eigenface feature space"),
        ("6. Classification", "SVM/KNN/Euclidean comparison determines the detected identity"),
    ]
    for i, (title, desc) in enumerate(steps):
        x = 0.8 + (i % 2) * 6.1
        y = 1.8 + (i // 2) * 1.55
        box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(5.4), Inches(1.15))
        box.fill.solid(); box.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
        box.line.color.rgb = BLUE
        tf = box.text_frame
        p1 = tf.paragraphs[0]; p1.text = title; p1.font.size = Pt(18); p1.font.bold = True; p1.font.color.rgb = NAVY
        p2 = tf.add_paragraph(); p2.text = desc; p2.font.size = Pt(15); p2.font.color.rgb = GRAY

    bottom = slide.shapes.add_shape(1, Inches(0.8), Inches(6.2), Inches(11.6), Inches(0.7))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "This makes the project a complete real-world use of Eigenfaces: detection + preprocessing + PCA + recognition."
    bottom.text_frame.paragraphs[0].font.size = Pt(17)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_9(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "7. Project results and performance", "What the system is designed to achieve")

    table = slide.shapes.add_shape(1, Inches(0.9), Inches(2.0), Inches(11.6), Inches(3.9))
    table.fill.solid(); table.fill.fore_color.rgb = LIGHT
    table.line.color.rgb = BLUE
    tf = table.text_frame
    tf.clear()
    rows = [
        ("Model", "Accuracy", "Macro F1", "Observation"),
        ("Euclidean", "68.25%", "0.6319", "Simple similarity-based matching"),
        ("KNN", "66.67%", "0.5997", "Strong local similarity but more errors"),
        ("SVM", "69.84%", "0.7385", "Best balance of performance and rejection"),
    ]
    for idx, row in enumerate(rows):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = " | ".join(row)
        p.font.size = Pt(17 if idx == 0 else 16)
        p.font.bold = idx == 0
        p.font.color.rgb = NAVY

    bottom = slide.shapes.add_shape(1, Inches(0.9), Inches(6.0), Inches(11.5), Inches(0.9))
    bottom.fill.solid(); bottom.fill.fore_color.rgb = NAVY
    bottom.line.fill.background()
    bottom.text = "In this project, Eigenfaces are combined with SVM classification and confidence filtering to support student recognition and unknown rejection."
    bottom.text_frame.paragraphs[0].font.size = Pt(18)
    bottom.text_frame.paragraphs[0].font.color.rgb = WHITE


def add_slide_10(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg_title(slide, "Conclusion", "Why Eigenfaces still matter")

    box = slide.shapes.add_shape(1, Inches(1.0), Inches(2.0), Inches(11.2), Inches(4.0))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = BLUE
    tf = box.text_frame
    for idx, item in enumerate([
        "Eigenfaces convert raw face images into a compact mathematical representation.",
        "The method learns the directions of maximum facial variation from the training set.",
        "This allows low-dimensional comparison, efficient recognition, and interpretable features.",
        "In this classroom system, Eigenfaces are used together with detection, preprocessing, and SVM classification to recognize students and reject unknowns.",
    ]):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(22)
        p.font.color.rgb = NAVY

    end = slide.shapes.add_textbox(Inches(1.2), Inches(6.1), Inches(8.0), Inches(0.7))
    end.text_frame.paragraphs[0].text = "Thank you!"
    end.text_frame.paragraphs[0].font.size = Pt(28)
    end.text_frame.paragraphs[0].font.bold = True
    end.text_frame.paragraphs[0].font.color.rgb = NAVY


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    make_assets()
    add_slide_1(prs)
    add_slide_2(prs)
    add_slide_3(prs)
    add_slide_4(prs)
    add_slide_5(prs)
    add_slide_6(prs)
    add_slide_7(prs)
    add_slide_8(prs)
    add_slide_9(prs)
    add_slide_10(prs)

    prs.save(OUT_PATH)
    print(f"Created: {OUT_PATH}")


if __name__ == "__main__":
    build_deck()
