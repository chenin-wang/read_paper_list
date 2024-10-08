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
                print("translatation failed.")


# 获取类别在文档中的插入位置
def get_category_insert_position(content, category):
    pattern = re.compile(rf"### {re.escape(category)}\n")
    match = pattern.search(content)
    if match:
        return match.end()  # 返回类别标题之后的插入点
    return None  # 如果类别不存在，返回None


# 提取现有类别下的最大编号
def get_last_index_for_category(content, category):
    pattern = re.compile(rf"#### (\d+).+\[{category}\]", re.IGNORECASE)
    matches = pattern.findall(content)
    if matches:
        return int(matches[-1])  # 返回该类别的最大编号
    return 0  # 如果没有找到，返回0


# 检查目录是否已存在
def toc_exists(content):
    return "<summary>Table of Contents</summary>" in content


# 将标题转为适合 GitHub Markdown 锚点格式的链接（小写、空格替换为-、去除特殊字符）
def generate_anchor(category):
    category = category.strip()  # 去除首尾空白
    category = category.replace("&", "and")  # 替换特殊字符
    category = re.sub(
        r"[^\w\s-]", "", category
    )  # 移除除字母、数字、连字符和空格外的字符
    return category.replace(" ", "-").lower()


# 检查类别是否已存在于目录中
def category_exists_in_toc(content, category):
    anchor = generate_anchor(category)
    toc_entry = f"<a href=#{anchor}>{category}</a>"
    return toc_entry in content


# 更新目录，若类别不存在则添加到目录中
def update_toc(content, category):
    if not category_exists_in_toc(content, category):
        toc_start_pos = content.find("<ol>", content.find("<details>"))
        toc_end_pos = content.find("</ol>", toc_start_pos)
        toc_entry = (
            f"    <li><a href=#{generate_anchor(category)}>{category}</a></li>\n"
        )
        content = content[:toc_end_pos] + toc_entry + content[toc_end_pos:]
    return content


# 更新 Markdown 内容
def update_markdown(new_links_by_category, translator):
    """更新 Markdown 内容"""
    filename = "README.md"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# arXiv 论文摘要\n\n"

    # 创建目录，并只创建一次
    toc_start = "<details>\n<summary>Table of Contents</summary>\n<ol>\n"
    toc_end = "</ol>\n</details>\n\n"
    if not toc_exists(content):
        toc_items = []
        for category in new_links_by_category:
            toc_items.append(
                f"    <li><a href=#{generate_anchor(category)}>{category}</a></li>"
            )
        toc = toc_start + "\n".join(toc_items) + toc_end
        content = toc + content

    # 遍历每个类别的新链接
    for category, links in new_links_by_category.items():
        # 更新目录
        content = update_toc(content, category)  # 更新目录

        # 获取类别最后一个编号
        last_index = get_last_index_for_category(content, category)
        current_index = last_index

        # 获取类别的插入位置
        insert_position = get_category_insert_position(content, category)

        # 如果类别不存在，添加到末尾，并插入类别标题
        if insert_position is None:
            new_content = f"\n### {category}\n\n"
            insert_position = len(content)  # 在文档末尾插入新类别
        else:
            new_content = ""

        # 处理每个链接
        for link in links:
            current_index += 1  # 递增编号
            arxiv_id = link.split("/")[-1]
            summary, title, publish_time = get_arxiv_summary(arxiv_id)
            logging.info(
                f"summary: {summary}, title: {title}, publish_time: {publish_time}"
            )

            # 创建翻译和图片下载的队列
            translation_queue = queue.Queue()
            image_queue = queue.Queue()

            # 定义线程函数
            def translation_thread():
                translated_summary = translate_to_chinese(summary, translator)
                translation_queue.put((arxiv_id, translated_summary))
                logging.info(f"Translation finished for {arxiv_id}")

            def image_download_thread():
                key_images = download_and_extract_key_images(arxiv_id)
                image_queue.put((arxiv_id, key_images))
                logging.info(f"Image download finished for {arxiv_id}")

            # 启动线程
            t1 = threading.Thread(target=translation_thread)
            t2 = threading.Thread(target=image_download_thread)
            t1.start()
            t2.start()

            # 等待线程结束
            t1.join()
            t2.join()

            # 从队列中获取结果
            trans_arxiv_id, translated_summary = translation_queue.get()
            img_arxiv_id, key_images = image_queue.get()

            # 确保结果是同一个 arxiv_id
            if trans_arxiv_id != img_arxiv_id or trans_arxiv_id != arxiv_id:
                logging.error(
                    f"Mismatch in arxiv_id for {arxiv_id}. Skipping this paper."
                )
                continue

            if translated_summary is None:
                logging.warning(f"Skipping {link} due to translation failure.")
                continue

            # 新增内容
            new_content += (
                f"#### {current_index}. [{title}]({link}) 发表时间: {publish_time}\n\n"
            )
            if key_images:
                for img_filename in key_images:
                    new_content += f"![Key Image]({img_filename})\n\n"
            new_content += f"{translated_summary}\n\n---\n\n"

        # 在指定位置插入新内容
        content = content[:insert_position] + new_content + content[insert_position:]

    # 写回文件
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)


def update_links_with_new_entries(links):
    previous_links = {}
    if os.path.exists("previous_links.json"):
        try:
            with open("previous_links.json", "r") as f:
                content = f.read().strip()
                if content:
                    previous_links = json.loads(content)
                else:
                    previous_links = {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding previous_links.json: {str(e)}")
            previous_links = {}

    # 整理出每个类别的新链接
    new_links_by_category = {}
    for category, link_list in links.items():
        new_links_by_category[category] = [
            link for link in link_list if link not in previous_links.get(category, [])
        ]

    # 按类别更新 markdown
    for category, links in new_links_by_category.items():
        if links:
            update_markdown({category: links}, translator)
            previous_links[category] = previous_links.get(category, []) + links

    # 更新 previous_links.json 文件
    with open("previous_links.json", "w") as f:
        json.dump(previous_links, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--google_api_key", type=str, required=True, help="Google AI API key."
    )
    args = parser.parse_args()
    os.makedirs("images", exist_ok=True)
    translator = Translater(api_key=args.google_api_key)

    try:
        with open("arxiv_links.json", "r") as f:
            links = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding arxiv_links.json: {str(e)}")
        links = {}

    update_links_with_new_entries(links)

    logging.info("Update completed successfully.")
