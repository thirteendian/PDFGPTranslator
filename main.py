import pymupdf, os, re
from tqdm import tqdm
import subprocess, tempfile, pathlib
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
load_dotenv()

import aiohttp, asyncio, hashlib, shelve


os.environ["TESSDATA_PREFIX"]= "/opt/homebrew/share/tessdata"
CACHE = shelve.open("trans_cache.db")          # md5(text) → english

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in .env file")
#The page should be read in order of 
# LEFT‑col  |  RIGHT‑col  |  ONE‑COL (full‑width stripes: foot‑notes, chapter titles, boxed remarks, top‑of‑page headers … anywhere on the page)

# deu fra
def ocr_pixmap_with_cli(pix, lang="deu", psm=3, dpi=300):
    """
    pix   PyMuPDF Pixmap
    Returns a bytes object containing a searchable 1 page PDF
    produced by Tesseract with the given options.
    """
    # temp folder, deleted when exiting "with"
    with tempfile.TemporaryDirectory() as td:
        png_path = pathlib.Path(td) / "page.png"
        pdf_path = pathlib.Path(td) / "page.pdf"

        pix.save(png_path)                        # write PNG

        # CLI call tesseract
        # Note that Non-CLI wrapper provides no benefits
        cmd = [
            "tesseract",
            str(png_path),
            str(pdf_path.with_suffix("")),        # Tesseract adds .pdf
            "-l", lang,
            "--dpi", str(dpi),
            "--psm", str(psm),
            "pdf"
        ]
        # add any key=value configs here
        cmd += ["-c", "preserve_interword_spaces=1"]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)

        return pdf_path.read_bytes()

ALNUM = re.compile(r"[A-Za-zÄÖÜäöüß0-9]")
def filter_noise(blocks, min_alpha=0.5, min_len=3):
    """Drop blocks that are mostly garbage (images, decorative lines …)."""
    good = []
    for b in blocks:
        txt = b[4].strip()
        if len(txt) < min_len:
            continue
        alpha_ratio = len(ALNUM.findall(txt)) / len(txt)
        if alpha_ratio < min_alpha:
            continue
        good.append(b)
    return good

def find_gutter(blocks):
    """Return x coordinate that best separates two columns."""
    xs = sorted(b[0] for b in blocks)            # left edges
    gaps = [(xs[i+1]-xs[i], xs[i], xs[i+1]) for i in range(len(xs)-1)]
    gap, x_left, x_right = max(gaps, key=lambda g:g[0])
    return (x_left + x_right) / 2

def classify_blocks_column_first(blocks, page_w, page_h,
                                 wide_ratio=0.65, head_band=0.25):
    """
    HEAD = wide blocks in the top head_band (titles, quotes)
    LEFT / RIGHT = decided by *centre‑distance* to column means
    FOOT = everything below an adaptive y_cut
    """

    # ── 0.  HEAD stripes ───────────────────────────────────────────────────
    HEAD = [b for b in blocks
            if (b[2]-b[0]) > page_w*wide_ratio and b[1] < page_h*head_band]
    body = [b for b in blocks if b not in HEAD]

    if len(body) < 2:
        return HEAD, body, [], []

    # ── 1.  adaptive y_cut via 1‑D k‑means (fast percentile variant) ───────
    ys = np.array(sorted(b[1] for b in body))
    gap_idx = np.argmax(np.diff(ys))             # largest vertical gap
    y_cut = ys[gap_idx] + 5                      # 5 pt margin

    # ── 2.  find column centres from blocks *above* y_cut ──────────────────
    narrow = [b for b in body if b[1] < y_cut and (b[2]-b[0]) <= page_w*wide_ratio]
    if len(narrow) < 2:                          # fallback to median split
        mean_L, mean_R = page_w*0.25, page_w*0.75
    else:
        cx = np.array(sorted((b[0]+b[2])/2 for b in narrow))
        mid = len(cx)//2
        mean_L = cx[:mid].mean()
        mean_R = cx[mid:].mean()

    # ── 3.  bucket blocks by centre‑distance ───────────────────────────────
    LEFT, RIGHT, FOOT = [], [], []
    for b in body:
        if b[1] >= y_cut:
            FOOT.append(b)
        else:
            centre = (b[0]+b[2]) / 2
            (LEFT if abs(centre - mean_L) < abs(centre - mean_R) else RIGHT).append(b)

    for lst in (HEAD, LEFT, RIGHT, FOOT):
        lst.sort(key=lambda b: b[1])

    return HEAD, LEFT, RIGHT, FOOT



def merge_footnotes(foot, gap=8):
    """Merge consecutive foot-note lines that belong to the same paragraph."""
    if not foot: return foot
    merged = [foot[0]]
    for b in foot[1:]:
        prev = merged[-1]
        if b[1] - prev[3] < gap:            # vertical touch → merge
            merged[-1] = [prev[0], prev[1], max(prev[2],b[2]), b[3],
                          prev[4].rstrip()+" "+b[4].lstrip()]
        else:
            merged.append(b)
    return merged

def looks_like_folio(block, page_w, page_h,
                     centre_tol=35, edge_tol=40):
    """
    Return True if *block* is probably just a page number.

    Heuristics
    ----------
    • Text is 1 to 3 digit integer (e.g. "1", "12", "327")
    • AND EITHER
         it is centred horizontally (± centre_tol points)
      OR
         it sits within edge_tol points of the top or bottom page edge
          (works for top-right or bottom-right folios)
    """
    x0, y0, x1, y1, text, *_ = block
    txt = text.strip()

    # 1. Must look like a small integer
    if not txt.isdigit() or len(txt) > 3:
        return False

    cx = (x0 + x1) / 2
    near_centre = abs(cx - page_w / 2) < centre_tol
    near_edge   = (y1 < edge_tol) or (y0 > page_h - edge_tol)

    return near_centre or near_edge


################################################################### GPT
async def deepseek_translate(texts, session, model="deepseek-chat"):
    """
    texts : list[str]  (≤ 4 000 chars total is safe)
    returns list[str]  translated 1‑for‑1
    """
    joined = "\n\n".join(texts)
    h      = hashlib.md5(joined.encode()).hexdigest()
    if h in CACHE:             # cheap disk cache
        return CACHE[h].split("\n\n")

    payload = {
        "model": model,
        "messages": [
            {"role":"system",
             "content":"You are a professional translator. Keep figure numbers,"
                       " math and music symbols. Translate the following into English. Do NOT add explanations."},
            {"role":"user", "content": joined}
        ],
        "temperature": 0.2
    }
    headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    url     = "https://api.deepseek.com/v1/chat/completions"

    async with session.post(url, json=payload, timeout=60, headers=headers) as r:
        data = await r.json()
        out  = data["choices"][0]["message"]["content"].strip()

    # ensure we return exactly len(texts) items -------------
    parts = [p.strip() for p in re.split(r"\n{2,}", out) if p.strip()]
    if len(parts) != len(texts):          # fallback: naive split by sentence count
        parts = re.split(r"(?<=[.!?])\s+", out, maxsplit=len(texts)-1)
    CACHE[h] = "\n\n".join(parts)
    return parts

def collect_sentences(blocks):
    SENT_END = re.compile(r"[.!?…»”]$")
    sents, rects = [], []
    buf, buf_rects = "", []
    for x0,y0,x1,y1,text,*_ in blocks:
        if buf: buf += " "
        buf += text.strip()
        buf_rects.append((x0,y0,x1,y1))
        if SENT_END.search(text.strip()):
            sents.append(buf); rects.append(buf_rects)
            buf, buf_rects = "", []
    if buf: sents.append(buf); rects.append(buf_rects)
    return sents, rects

def even_split(text: str, n: int):
    """
    Split *text* into *n* segments with nearly equal word counts.
    Keeps word order; never breaks a word.
    """
    words = text.split()
    if n <= 1 or len(words) <= n:
        return [" ".join(words[i:i+1]) for i in range(len(words))] + [""]*(n-len(words))
    base = len(words) // n
    extra = len(words) % n
    out, idx = [], 0
    for i in range(n):
        take = base + (1 if i < extra else 0)
        out.append(" ".join(words[idx:idx+take]))
        idx += take
    return out
################################################################### MAIN


doc = pymupdf.open("test2.pdf")
out_doc = pymupdf.open()
# out = open("output.txt","wb")
for pno, page in tqdm(enumerate(doc),total=len(doc), desc="Progressing Pages"): # iterate the document pages

    # gets the visible rectangle(with rotation info)
    ## For page rect without roatation info, see mediabox and cropbox
    rect = page.rect

    #page.get_pixmap will take page like screenshoot, for extract embedded images refer to pymupdf.Pixmap(doc, xref) method
    pix = page.get_pixmap(dpi=300, clip=rect, alpha=False)

    ocr_pdf_bytes = ocr_pixmap_with_cli(pix,"deu",psm=3,dpi=300)
    #ocr_bytes = pix.pdfocr_tobytes(compress=True, language='deu', options=opts)

    
    # open this in memory 1 OCRed-page
    ocr_doc = pymupdf.open("pdf", ocr_pdf_bytes) 

    # the text extraction is at ocr_doc[0]
    ocr_page = ocr_doc[0]
    w, h = ocr_page.rect.width, ocr_page.rect.height 

    blocks = ocr_page.get_text("blocks")
    
    # 1. remove noise and drop image false recognization
    blocks = filter_noise(blocks)
    # 2. remove page numbers
    blocks = [b for b in blocks if not looks_like_folio(b, w, h)]
    # 3. classify
    head, left, right, foot = classify_blocks_column_first(blocks, w, h)
    # 4. merge foot-note if seperate by the OCR
    foot = merge_footnotes(foot)
    # 5. order everything
    ordered_blocks = head + left + right + foot

    ##############
    sentences, rect_map = collect_sentences(ordered_blocks)

    # --- batch translate (<= 50 sentences per request keeps tokens small) ----
    async def translate_page(sentences):
        out = []
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(sentences), 50):
                chunk = sentences[i:i+50]
                out.extend(await deepseek_translate(chunk, session))
        return out

    english = asyncio.run(translate_page(sentences))

    # 1. new blank page with same size
    new_page = out_doc.new_page(width=w, height=h)

    # 2. background = original scan
    new_page.insert_image(new_page.rect, pixmap=pix, overlay=False)

    # 3. overlay English text
    tw = pymupdf.TextWriter(new_page.rect)
    for en, rects in zip(english, rect_map):
        segs = even_split(en, len(rects)) 
        for seg, (x0,y0,x1,y1) in zip(segs, rects):
            r   = pymupdf.Rect(x0, y0, x1, y1)
            fsz = max(6, 0.7 * r.height)
            tw.fill_textbox(r, seg, fontsize=fsz,
                            align=pymupdf.TEXT_ALIGN_LEFT)

    tw.write_text(new_page, overlay=True)


    # --- append translated page to an output PDF ------------------------------
    # if pno == 0:
    #     out_doc = pymupdf.open()               # create once
    out_doc.insert_pdf(ocr_doc, from_page=0, to_page=0, start_at=pno)
        # with open("blocks.txt", "a", encoding="utf-8") as f:
        #     f.write(f"\n--- page {pno+1} ---\n")
        #     for blkno, blk in enumerate(ordered_blocks):
        #          f.write(f"[block{blkno+1}]:"+blk[4].strip() + "\n")
out_doc.save("test2_en.pdf", deflate=True, garbage=4)
print("Done → test2_en.pdf")

