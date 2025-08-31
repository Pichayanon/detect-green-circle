from pdf2image import convert_from_path
import os
pdf_path = 'pdf/34. Vanguard S&P 500 Growth Index Fund (VOOG).pdf'
output_dir = 'pdf_pages_VanEck'

os.makedirs(output_dir, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=300)
for i, page in enumerate(pages):
    out_path = os.path.join(output_dir, f'page_{i+1}.png')
    page.save(out_path, 'PNG')
    print(f'Saved {out_path}')
