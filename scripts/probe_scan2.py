"""生成第二份扫描件测试 PDF（内容不在知识库中，用于完整入库验证）。"""

import fitz

SRC = "研究系统方案推荐 .pdf"
DST = "_test_scan_plan.pdf"

src = fitz.open(SRC)
out = fitz.open()
for pno in range(min(2, src.page_count)):
    page = src[pno]
    pix = page.get_pixmap(dpi=150)
    img = fitz.open()
    img_page = img.new_page(width=page.rect.width, height=page.rect.height)
    img_page.insert_image(img_page.rect, pixmap=pix)
    out.insert_pdf(img)
out.save(DST)
chk = fitz.open(DST)
print(
    "generated:",
    DST,
    "pages:",
    chk.page_count,
    "native text chars p1:",
    len(chk[0].get_text().strip()),
)
