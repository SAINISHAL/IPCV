"""Safe archive extractor and dataset pair resolver."""
import os
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from src.utils.logger import setup_logger

logger = setup_logger("ArchiveExtractor")


class ArchiveExtractor:
    """Safely extracts ZIP archives and matches image-annotation pairs."""

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, archives_dir: str or Path, output_raw_dir: str or Path):
        self.archives_dir = Path(archives_dir)
        self.output_raw_dir = Path(output_raw_dir)
        self.output_raw_dir.mkdir(parents=True, exist_ok=True)

    def extract_all(self, overwrite: bool = False) -> Dict[str, Path]:
        """Safely extract all student zip archives into output_raw_dir.
        
        Returns:
            Dictionary mapping student archive name -> extracted directory path.
        """
        extracted = {}
        zip_files = sorted(list(self.archives_dir.glob("*.zip")))
        
        for zf in zip_files:
            # Skip code repositories like Real_Time_Face_Recognition_System-master.zip
            if "Real_Time" in zf.name or "eigenfaces" in zf.name:
                continue
                
            clean_name = zf.stem.split("-2026")[0].strip()
            target_dir = self.output_raw_dir / clean_name
            
            if target_dir.exists() and not overwrite:
                logger.info(f"Archive '{clean_name}' already extracted at {target_dir}")
                extracted[clean_name] = target_dir
                continue
                
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting '{zf.name}' -> '{target_dir}'...")
            
            try:
                with zipfile.ZipFile(zf, "r") as z:
                    for member in z.infolist():
                        # Protect against Zip Slip directory traversal
                        dest_path = (target_dir / member.filename).resolve()
                        if not str(dest_path).startswith(str(target_dir.resolve())):
                            logger.warning(f"Unsafe file path detected in zip: {member.filename}")
                            continue
                        if member.is_dir():
                            dest_path.mkdir(parents=True, exist_ok=True)
                        else:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with z.open(member) as src, open(dest_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                                
                extracted[clean_name] = target_dir
            except Exception as e:
                logger.error(f"Failed to extract {zf.name}: {e}")
                
        return extracted

    @classmethod
    def resolve_pairs(cls, student_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
        """Pair image files with their corresponding YOLO annotation text files.
        
        Handles tricky naming conventions:
          - 'image 1.jpg' <-> 'image 1.txt'
          - '01_photo.jpg' <-> '02_annotation.txt'
          - 'img.jpg.png' <-> 'img.jpg.txt'
        """
        all_files = list(student_dir.rglob("*"))
        images = [f for f in all_files if f.is_file() and f.suffix.lower() in cls.IMAGE_EXTS]
        texts = [f for f in all_files if f.is_file() and f.suffix.lower() == ".txt"]
        
        pairs = []
        used_texts = set()

        # Step 1: Direct matching by stem or base name
        for img in images:
            img_stem_clean = img.stem.replace(" ", "").lower()
            matched_txt = None
            
            for txt in texts:
                if txt in used_texts:
                    continue
                txt_stem_clean = txt.stem.replace(" ", "").replace("_annotation", "").replace(".jpg", "").lower()
                if img_stem_clean == txt_stem_clean or txt_stem_clean in img_stem_clean or img_stem_clean in txt_stem_clean:
                    matched_txt = txt
                    used_texts.add(txt)
                    break
                    
            # Step 2: Handle odd-even sequential numbering (e.g. 01_photo vs 02_annotation)
            if matched_txt is None and ("photo" in img_stem_clean or img_stem_clean.isdigit()):
                digits = "".join([c for c in img.stem if c.isdigit()])
                if digits:
                    num = int(digits)
                    target_txt_num = num + 1  # 01 -> 02, 03 -> 04, etc.
                    for txt in texts:
                        if txt in used_texts:
                            continue
                        txt_digits = "".join([c for c in txt.stem if c.isdigit()])
                        if txt_digits and int(txt_digits) == target_txt_num:
                            matched_txt = txt
                            used_texts.add(txt)
                            break
                            
            pairs.append((img, matched_txt))
            
        return pairs

