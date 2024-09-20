import json
import requests
from PIL import Image
import os
import google.generativeai as genai
import time
import argparse
import logging
import arxiv
import threading
import queue
import fitz  # PyMuPDF
import io
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Translater:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)  # 填入自己的api_key

        # 查询模型
        for m in genai.list_models():
            print(m.name)
            print(m.supported_generation_methods)
        sys_prompt = (
            "You are a highly skilled translator specializing in artificial intelligence and computer science. \
            You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.\
            Your translations are known for their accuracy, clarity, and fluency.\n\
            Your task is to translate technical academic abstracts from English to Simplified Chinese.\
            You will receive an English abstract, and you should produce a Chinese translation that adheres to the following:\n\
            * **Accuracy:** All technical terms and concepts must be translated correctly.\n\
            * **Clarity:** The translation should be easily understood by someone familiar with AI concepts.\n\
            * **Fluency:** The translation should read naturally in Chinese.\n\
            * **Output Format:** The returned text should not be bolded, not be separated into paragraphs, and remove all line breaks to merge into a single paragraph.\n \
            Do not add your own opinions or interpretations; remain faithful to the original text while optimizing for readability. \
            "
        )

        self.model = genai.GenerativeModel(
            "gemini-1.5-pro-latest",
            system_instruction=sys_prompt,
            generation_config=genai.GenerationConfig(
                # max_output_tokens=2000,
                temperature=0.8,
            ),
        )

    # models/gemini-pro
    # 输入令牌限制:30720
    # 输出令牌限制:2048
    # 模型安全:自动应用的安全设置，可由开发者调整。如需了解详情，请参阅安全设置

    def translate(self, text: str):
        response = self.model.generate_content(
            f"Note output format, here is the abstract to translate:\n{text}"
        )
        return response.text


def get_arxiv_summary(arxiv_id):
    # url = "https://arxiv.org/abs/2406.00428"
    try:
        paper = next(arxiv.Search(id_list=[arxiv_id]).results())
        summary = paper.summary.strip()
        title = paper.title.strip()
        publish_time = paper.published.date()
        # paper_abstract = paper.summary.replace("\n", " ")
        # paper_authors = get_authors(paper.authors)
        # paper_first_author = get_authors(paper.authors, first_author=True)
        # primary_category = paper.primary_category
        # update_time = paper.updated.date()
        # comments = paper.comment
        return summary, title, publish_time
    except StopIteration:
        logging.error(f"No paper found for arXiv ID: {arxiv_id}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error fetching arXiv summary for {arxiv_id}: {str(e)}")
        return None, None, None


import os
import re
import requests
import fitz  # PyMuPDF
import logging


def recoverpix(doc, item):
    xref, smask = item[0], item[1]  # xref of PDF image and its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        pix0 = (
            fitz.Pixmap(pix0, 0) if pix0.alpha else pix0
        )  # remove alpha channel if present
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

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

    return doc.extract_image(xref)


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
                    r"architectures|architecture|pipeline|pipelines|framework|structure",
                    text_around,
                    re.IGNORECASE,
                ):
                    image = recoverpix(doc, img)
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


def translate_to_chinese(text, translator):
    logging.info(f"Translating: {text[:50]}...")  # Log first 50 characters
    retry_count = 0
    retry_seconds = 60
    NUM_RETRIES = 3
    while retry_count < NUM_RETRIES:
        try:
            paper_abstract = translator.translate(text)
            return paper_abstract
        except Exception as e:
            logging.error(
                f"Translation error: {e}. Retrying in {retry_seconds} seconds."
            )
            time.sleep(retry_seconds)
            retry_count += 1
            # Here exponential backoff is employed to ensure the account doesn't get rate limited by making
            # too many requests too quickly. This increases the time to wait between requests by a factor of 2.
            retry_seconds *= 2
        finally:
            if retry_count == NUM_RETRIES:
                print("Could not recover after making " f"{retry_count} attempts.")
                print(f"translatation failed. paper_abstract:\n {paper_abstract} ")
    return translator.translate(text, dest="zh-cn").text


def update_markdown(new_links, translator):
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# arXiv 论文摘要\n\n"

    new_content = ""
    for link in new_links:
        arxiv_id = link.split("/")[-1]
        summary, title, publish_time = get_arxiv_summary(arxiv_id)
        logging.info(
            f"summary: {summary}, title: {title}, publish_time: {publish_time}"
        )
        # Create queues for translation and image download results
        translation_queue = queue.Queue()
        image_queue = queue.Queue()

        # Define thread functions
        def translation_thread():
            translated_summary = translate_to_chinese(summary, translator)
            translation_queue.put((arxiv_id, translated_summary))
            logging.info(f"Translation finished for {arxiv_id}")

        def image_download_thread():
            key_images = download_and_extract_key_images(arxiv_id)
            image_queue.put((arxiv_id, key_images))
            logging.info(f"Image download finished for {arxiv_id}")

        # Start threads
        t1 = threading.Thread(target=translation_thread)
        t2 = threading.Thread(target=image_download_thread)
        t1.start()
        t2.start()

        # Wait for both threads to finish
        t1.join()
        t2.join()

        # Get results from queues
        trans_arxiv_id, translated_summary = translation_queue.get()
        img_arxiv_id, key_images = image_queue.get()

        # Ensure the results are for the same arxiv_id
        if trans_arxiv_id != img_arxiv_id or trans_arxiv_id != arxiv_id:
            logging.error(f"Mismatch in arxiv_id for {arxiv_id}. Skipping this paper.")
            continue

        if translated_summary is None:
            logging.warning(f"Skipping {link} due to translation failure.")
            continue

        new_content += f"## [{title}]({link}) 发表时间: {publish_time}\n\n"
        if key_images:
            for img_filename in key_images:
                new_content += f"![Key Image]({img_filename})\n\n"
        else:
            logging.warning(f"No images found for {arxiv_id}")
        new_content += f"{translated_summary}\n\n---\n\n"

    if new_content:
        content = new_content + content
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)
    else:
        logging.info("No new content to update in README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--google_api_key", type=str, required=True, help="Google AI API key."
    )
    args = parser.parse_args()

    translator = Translater(api_key=args.google_api_key)

    try:
        with open("arxiv_links.json", "r") as f:
            links = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding arxiv_links.json: {str(e)}")
        links = []

    os.makedirs("images", exist_ok=True)

    previous_links = []
    if os.path.exists("previous_links.json"):
        try:
            with open("previous_links.json", "r") as f:
                content = f.read().strip()
                if content:
                    previous_links = json.loads(content)
                else:
                    previous_links = []
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding previous_links.json: {str(e)}")
            previous_links = []

    new_links = [link for link in links if link not in previous_links]

    update_markdown(new_links, translator)

    with open("previous_links.json", "w") as f:
        json.dump(links, f)

    logging.info("Update completed successfully.")
