import os

from pdf2image import convert_from_path

pdf_path = 'pdf/38. Vanguard Small Cap Growth Index Fund (VBK).pdf'
output_dir = 'pdf_page_output'

os.makedirs(output_dir, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=300)
for i, page in enumerate(pages):
    out_path = os.path.join(output_dir, f'page_{i+1}.png')
    page.save(out_path, 'PNG')
    print(f'IMAGE PATH:  {out_path}')
