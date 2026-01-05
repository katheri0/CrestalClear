from flask import Flask, request, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import io
import re

from correctionIllumination import normalizeDocumentIllumination
from noiseReduction import reduceDocumentNoise
from binarization import binarizeDocumentImage


app = Flask(__name__)


def processSingleDocumentImage(documentImage: np.ndarray) -> np.ndarray:
    normalizedImage = normalizeDocumentIllumination(documentImage)
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


@app.route("/", methods=["GET", "POST"])
def uploadImages():
    if request.method == "POST":
        uploadedFiles = request.files.getlist("images")
        rawPdfName = request.form.get("pdf_name", "")
        safePdfName = sanitizeFilename(rawPdfName)

        processedImages: list[np.ndarray] = []

        for uploadedFile in uploadedFiles:
            fileBytes = np.frombuffer(uploadedFile.read(), np.uint8)
            documentImage = cv2.imdecode(fileBytes, cv2.IMREAD_COLOR)

            if documentImage is None:
                continue

            processedImage = processSingleDocumentImage(documentImage)
            processedImages.append(processedImage)

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
