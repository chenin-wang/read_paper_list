import os
import re
import requests
import fitz  # PyMuPDF
import logging
from PIL import Image, ImageDraw


def recoverpix(doc, item):
    xref, smask = item[0], item[1]  # xref of PDF image and its /SMask

    try:
        # special case: /SMask or /Mask exists
        if smask > 0:
            pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
            pix0 = (
                fitz.Pixmap(pix0, 0) if pix0.alpha else pix0
            )  # remove alpha channel if present
            mask = fitz.Pixmap(doc.extract_image(smask)["image"])

            pix = fitz.Pixmap(pix0, mask)
            ext = "pam" if pix0.n > 3 else "png"
            return {
                "ext": ext,
                "colorspace": pix.colorspace.n,
                "image": pix.tobytes(ext),
            }

        # special case: /ColorSpace definition exists
        if "/ColorSpace" in doc.xref_object(xref, compressed=True):
            pix = fitz.Pixmap(doc, xref)
            pix = fitz.Pixmap(fitz.csRGB, pix)
            return {
                "ext": "png",
                "colorspace": 3,
                "image": pix.tobytes("png"),
            }

        return doc.extract_image(xref), doc.get_image_bbox(xref)
    except Exception as e:
        logging.error(f"Error recovering pixel data for xref {xref}: {e}")
        return None


def draw_graphics(graphics_data, page_width, page_height):
    # 创建空白图像
    img = Image.new("RGB", (int(page_width), int(page_height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for item in graphics_data:
        # 处理填充形状
        if item["type"] == "fs":
            rect = item["rect"]
            fill_color = tuple(int(c * 255) for c in item["fill"])  # 转换为 RGB 格式
            draw.rectangle([rect.x0, rect.y0, rect.x1, rect.y1], fill=fill_color)

        # 处理其他图形类型...

    return img


def download_and_extract_key_images(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = f"{arxiv_id}.pdf"

    # Download PDF
    with open(pdf_path, "wb") as file:
        file.write(requests.get(url).content)

    # Open PDF and create images directory
    doc = fitz.open(pdf_path)
    page_count = doc.page_count  # number of pages
    os.makedirs("images", exist_ok=True)

    key_images = []
    dimlimit = 100  # Each image side must be greater than this
    relsize = 0.05  # Image size ratio must be larger than this (5%)
    abssize = 2048  # Absolute image size limit (2 KB): ignore if smaller

    for page_num in range(page_count):
        # graphics_data = doc[page_num].get_drawings()
        # # 从 `graphics_data` 中获取页面宽度和高度
        # page_width = max(item["rect"].x1 for item in graphics_data)
        # page_height = max(item["rect"].y1 for item in graphics_data)

        # # 绘制图形并保存图像
        # img = draw_graphics(graphics_data, page_width, page_height)
        # img.save(f"extracted_graphics{page_num}.png")

        il = doc.get_page_images(page_num)
        for image_index, img in enumerate(il):
            try:
                bbox = doc[page_num].get_image_bbox(img[7])
                # Extract text around the image
                expand_by = 50
                expanded_rect = fitz.Rect(
                    bbox.x0 - expand_by,
                    bbox.y0 - expand_by,
                    bbox.x1 + expand_by,
                    bbox.y1 + expand_by,
                )
                text_around = doc[page_num].get_text("text", clip=expanded_rect)

                # Check if the text contains "architecture" or "pipeline"
                if re.search(
                    r"architectures|architecture|pipeline|pipelines",
                    text_around,
                    re.IGNORECASE,
                ):
                    image = recoverpix(doc, img)
                    if image is None:
                        continue  # Skip if image recovery failed

                    n = image["colorspace"]
                    imgdata = image["image"]

                    # Validate image dimensions and size
                    width = img[2]
                    height = img[3]
                    if (
                        min(width, height) > dimlimit
                        and len(imgdata) > abssize
                        and len(imgdata) / (width * height * n) > relsize
                    ):
                        image_filename = f"images/{arxiv_id}_page_{page_num+1}_img_{image_index+1}.{image['ext']}"
                        with open(image_filename, "wb") as fout:
                            fout.write(imgdata)

                        key_images.append(image_filename)
                        logging.info(f"Key image saved as {image_filename}")

            except Exception as e:
                logging.error(
                    f"Could not process image on page {page_num + 1}, image index {image_index}: {e}"
                )

    # Clean up
    doc.close()
    os.remove(pdf_path)

    return key_images


if __name__ == "__main__":
    arxiv_id = "2103.14030"
    download_and_extract_key_images(arxiv_id)
