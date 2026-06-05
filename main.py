from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import io
import re
import concurrent.futures

from correctionIllumination import normalizeDocumentIllumination
from noiseReduction import reduceDocumentNoise
from binarization import binarizeDocumentImage
from geometricNormalization import deskewDocumentImage


app = Flask(__name__)


def processSingleDocumentImage(documentImage: np.ndarray) -> np.ndarray:
    deskewedImage = deskewDocumentImage(documentImage)
    normalizedImage = normalizeDocumentIllumination(deskewedImage)
    denoisedImage = reduceDocumentNoise(normalizedImage)
    binaryImage = binarizeDocumentImage(denoisedImage)
    return binaryImage


def convertImagesToPdf(processedImages: list[np.ndarray]) -> io.BytesIO:
    pilImages: list[Image.Image] = []

    for processedImage in processedImages:
        pilImage = Image.fromarray(processedImage).convert("L")
        pilImages.append(pilImage)

    pdfBuffer = io.BytesIO()
    pilImages[0].save(
        pdfBuffer,
        format="PDF",
        save_all=True,
        append_images=pilImages[1:]
    )
    pdfBuffer.seek(0)
    return pdfBuffer


def sanitizeFilename(rawFilename: str) -> str:
    """
    Allow only letters, numbers, dash, underscore.
    Prevent empty or dangerous filenames.
    """
    cleanedFilename = re.sub(r"[^a-zA-Z0-9_-]", "", rawFilename)

    if not cleanedFilename:
        return "document"

    return cleanedFilename


def decode_and_process(fileBytesRaw):
    fileBytes = np.frombuffer(fileBytesRaw, np.uint8)
    documentImage = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)
    if documentImage is None:
        return None

    # Cap image resolution to a maximum of 2500 pixels on the longest side
    max_allowed_dim = 2500
    h, w = documentImage.shape[:2]
    if max(h, w) > max_allowed_dim:
        scale = max_allowed_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        documentImage = cv2.resize(
            documentImage, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return processSingleDocumentImage(documentImage)


@app.route("/", methods=["GET", "POST"])
def uploadImages():
    if request.method == "POST":
        uploadedFiles = request.files.getlist("images")
        rawPdfName = request.form.get("pdf_name", "")
        safePdfName = sanitizeFilename(rawPdfName)

        file_bytes_list = []
        for uf in uploadedFiles:
            b = uf.read()
            if b:
                file_bytes_list.append(b)

        processedImages: list[np.ndarray] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(decode_and_process, file_bytes_list))

        processedImages = [img for img in results if img is not None]

        if not processedImages:
            return "No valid images uploaded", 400

        pdfBuffer = convertImagesToPdf(processedImages)

        return send_file(
            pdfBuffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{safePdfName}.pdf"
        )

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
