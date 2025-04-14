# PDFGPTranslator
A translator using GPT api, while preserving the original format

```
Here is a translation process from German to English:
          ┌──────────┐      ┌──────────┐      ┌─────────────┐
PDF  ───► │  Split   │ ───► │  OCR &   │ ───► │  Translate  │
(scanned) │  pages   │      │  cleanup │      │  (GPT‑4o)   │
          └──────────┘      └──────────┘      └─────────────┘
               │                  │                  │
               ▼                  ▼                  ▼
          images + XML      clean German       clean English
               │                                   │
               └─────────────┬─────────────────────┘
                             ▼
                      ┌────────────┐
                      │  Rebuild   │
                      │   PDF      │
                      └────────────┘
```  
(convert to image) -> 
OCR with bounding boxes(hOCR with XML) ->
convert hOCR to PDF points&group paragraphs ->
translate each block(api) ->
re-assemble PDF



# For Page Split & OCR
`pyMuPDF` is the most suitable. We don't even need to convert the document to image first, it support OCR on document directly.

`pyMuPDF` + `Tesseract` is the most suitable. 
If we use external Tesseract, we need extra step to convert and regroup by multiple 72/dpi(because 1pt~1/72 inch) to get pdf point grid(pt) from px. pyMuPDF have direct image/pdf OCR with embedded Tesseract, which will get object directly.

To use embedded Tesseract of pyMuPDF, `brew` provide packages to download it's language model, using 
```
brew install tesseract-lang
```
The address can be checked through `brew info`, and set this tessdata folder to the environment variable `TESSDATA_PREFIX`.


## How to keep the original arrangement

Every single page as background bitmap and lay a brand-new English text on top of it at the exact same rectangles that OCR detected.

# Translation

# PDF recreatiion



