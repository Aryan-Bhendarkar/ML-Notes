"""
Batch OCR all slides in a lecture folder using EasyOCR.
Saves per-slide text and a combined output.
"""
import easyocr
from PIL import Image
import os, sys, time, tempfile

def ocr_lecture_folder(slide_dir, output_path):
    """OCR all slide_NNN.jpg files in slide_dir, save to output_path."""
    tmpdir = tempfile.gettempdir()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    
    # Find all slide images
    slides = sorted([f for f in os.listdir(slide_dir) if f.startswith('slide_') and f.endswith('.jpg')])
    
    all_text = []
    total_time = 0
    
    for i, slide_file in enumerate(slides):
        slide_path = os.path.join(slide_dir, slide_file)
        img = Image.open(slide_path)
        # Resize to half for speed
        img_small = img.resize((960, 529))
        tmp_path = os.path.join(tmpdir, f'ocr_temp_{i}.jpg')
        img_small.save(tmp_path)
        
        start = time.time()
        result = reader.readtext(tmp_path, detail=0, paragraph=True)
        elapsed = time.time() - start
        total_time += elapsed
        
        slide_text = '\n'.join(result)
        all_text.append(f"=== {slide_file} (slide {i+1}/{len(slides)}) ===\n{slide_text}\n")
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(slides)} slides ({elapsed:.1f}s each)")
        
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass
    
    # Write combined output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# OCR Output: {os.path.basename(slide_dir)}\n")
        f.write(f"# Total slides: {len(slides)}\n")
        f.write(f"# Total time: {total_time:.1f}s\n\n")
        f.write('\n'.join(all_text))
    
    print(f"\nDone: {len(slides)} slides OCR'd in {total_time:.1f}s")
    print(f"Output saved to: {output_path}")
    return output_path

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python ocr_slides.py <slide_dir> <output_path>")
        sys.exit(1)
    
    slide_dir = sys.argv[1]
    output_path = sys.argv[2]
    ocr_lecture_folder(slide_dir, output_path)
