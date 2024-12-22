import os
import requests
import json
import re
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image, ImageOps
import io
import logging
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()

# Flask app configuration
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["EXTRACTED_IMAGES"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "extracted_images"
)

# Ensure upload and extracted_images directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["EXTRACTED_IMAGES"], exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# XML Formatting Functions
def parse_and_format_xml(input_xml_string):
    """Parse and format XML string to match desired output format."""
    try:
        # Parse the input XML string
        root = ET.fromstring(input_xml_string)

        # Extract publication details dynamically
        publication_name = root.find(".//publication/name").text if root.find(".//publication/name") is not None else "Saskal Agrowon"
        publication_code = root.find(".//publication/code").text if root.find(".//publication/code") is not None else "SAA"
        publication_date = root.find(".//publication/date").text if root.find(".//publication/date") is not None else "2024-09-04"

        # Create the new root element for the output XML
        zissor = ET.Element('zissor', {'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
        publication = ET.SubElement(zissor, 'publication', id="96")

        # Add dynamically fetched fields
        ET.SubElement(publication, 'name').text = publication_name
        ET.SubElement(publication, 'code').text = publication_code

        # Create the issue element
        issue = ET.SubElement(publication, 'issue', id="2413")
        ET.SubElement(issue, 'date').text = publication_date
        ET.SubElement(issue, 'edition').text = "0"

        # Create the section element
        section = ET.SubElement(issue, 'section')

        # Process the articles in the input XML
        article_sequence = 1
        for article in root.findall(".//article"):
            # Create a page element
            page = ET.SubElement(section, 'page', name="1", sequence="1", width="3177", height="4062")

            # Create an article element with incremental sequence
            article_elem = ET.SubElement(page, 'article', id=f"65357{article_sequence}", sequence=str(article_sequence))
            article_sequence += 1

            # Add title
            heading = article.find("heading").text if article.find("heading") is not None else "No Title"
            ET.SubElement(article_elem, 'title').text = heading

            # Add geometry with dynamic positioning
            geometry = ET.SubElement(article_elem, 'geometry')
            ET.SubElement(geometry, 'box', 
                         id=f"1007885{article_sequence}", 
                         hpos=str(112 + (article_sequence - 1) * 50),
                         vpos=str(758 + (article_sequence - 1) * 100),
                         width="1005",
                         height="60",
                         pagesequence="1",
                         pagename="1")

            # Add zonegroup for the heading
            zonegroup_heading = ET.SubElement(article_elem, 'zonegroup', type="1", sequence="1", typename="Heading")
            ET.SubElement(zonegroup_heading, 'zone', type="3", sequence="1", typename="Heading").text = f"<b>{heading}</b>"

            # Add content zones
            content = article.find("content").text if article.find("content") is not None else "No Content"
            paragraph_zonegroup = ET.SubElement(article_elem, 'zonegroup', type="4", sequence="2", typename="Paragraph")
            ET.SubElement(paragraph_zonegroup, 'zone', type="1", sequence="5", typename="Text").text = f"<b>{content}</b>"

            # Add image zonegroup if image exists
            image = article.find("image")
            if image is not None and image.text:
                image_zonegroup = ET.SubElement(article_elem, 'zonegroup', type="5", sequence="3", typename="Image")
                ET.SubElement(image_zonegroup, 'zone', type="2", sequence="6", typename="Image").text = image.text

        # Convert the XML tree to string
        return ET.tostring(zissor, encoding="unicode", xml_declaration=True)
    except Exception as e:
        logging.error(f"Error formatting XML: {e}")
        return None


# Helper functions
def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_blocks(pdf_path):
    """Extracts text blocks from a PDF using PyMuPDF."""
    document = fitz.open(pdf_path)
    all_text_blocks = []
    for page_number in range(len(document)):
        page = document[page_number]
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, block_type = block[:6]
            all_text_blocks.append(
                {
                    "page": page_number + 1,
                    "bbox": (x0, y0, x1, y1),
                    "text": text.strip(),
                    "type": block_type,
                }
            )
    document.close()
    return all_text_blocks


def prepare_prompt(text_blocks, images_with_captions):
    """Prepares a prompt for the LLM to structure the text blocks into XML format."""
    extracted_text = "\n".join(
        [block["text"] for block in text_blocks if block["text"]]
    )

    # Include image captions in the prompt
    image_info = "\n".join(
        [
            f"Image: {img['caption']} (Path: {img['path']})"
            for img in images_with_captions
        ]
    )

    prompt = f"""
    You are an expert in analyzing text and structuring it into XML format.
    Below is raw text extracted from a Marathi newspaper PDF and information about images with captions.
    Your task is to:
    - Please correct the text and then proceed with further processing as requested.
    - Identify the newspaper name and publication date.
    - Structure the articles into an XML format as follows:

    <publication>
      <name></name>
      <date></date>
      <section>
        <article>
          <heading></heading>
          <subheading></subheading>
          <author></author>
          <content></content>
          <image></image> <!-- Include the image path that matches the article, or leave it blank if no image matches. -->
        </article>
      </section>
    </publication>

    Rules:
    - Match the image captions with articles and include the corresponding image path in the <image> tag.
    - If no match is found for an article, leave the <image> tag blank.
    - Extract at least 20 words of content for each article.
    - Exclude articles with less than 20 words in the content.
    - Leave <subheading> and <author> blank if not available.
    - Only output valid XML without any extra text or comments.

    Raw Text:
    {extracted_text}

    Images with Captions:
    {image_info}
    """
    return prompt


def call_llm(prompt, api_key):
    """Sends the prepared prompt to Llama3.1-8b-instant to generate XML."""
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    model = "llama-3.1-70b-versatile"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        xml_output = response_json["choices"][0]["message"]["content"]
        return xml_output
    else:
        logging.error("Error in API Call:", response.text)
        return None


def sanitize_xml_output(xml_output):
    """Sanitizes the LLM output to retain only valid XML tags."""
    xml_content = re.search(r"<publication>.*</publication>", xml_output, re.DOTALL)
    if xml_content:
        return xml_content.group(0).strip()
    else:
        logging.error("No valid XML content found in the response.")
        return None


def find_caption_below_image(page, image_rect):
    """Find captions strictly below the image based on its position."""
    margin = 20
    try:
        blocks = page.get_text("blocks")
        for block in blocks:
            block_rect = fitz.Rect(block[:4])

            is_below = (
                block_rect.y0 >= image_rect.y1
                and abs(block_rect.y0 - image_rect.y1) < margin
                and block_rect.x0 >= image_rect.x0 - margin
                and block_rect.x1 <= image_rect.x1 + margin
            )

            if is_below:
                text = block[4].strip()
                if text and len(text) > 10:
                    return text
    except Exception as e:
        logging.error(f"Error finding caption below image: {e}")
    return None


def correct_inverted_image(image):
    """Detect and correct color-inverted images."""
    try:
        if image.mode in ["1", "L", "P"]:
            image = image.convert("RGB")
        elif image.mode == "CMYK":
            image = ImageOps.invert(image.convert("RGB"))
        return image
    except Exception as e:
        logging.error(f"Error correcting inverted image: {e}")
        return image


def extract_images_with_captions(pdf_path):
    """Extract images from the PDF with captions strictly below them."""
    images = []
    try:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(app.config["EXTRACTED_IMAGES"], f"{pdf_name}_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        doc = fitz.open(pdf_path)
        image_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if base_image:
                    image_rects = page.get_image_rects(xref)
                    if not image_rects:
                        continue

                    image_rect = image_rects[0]
                    caption = find_caption_below_image(page, image_rect)

                    if caption:
                        image_count += 1
                        image_filename = f"{pdf_name}_image{image_count}.png"
                        image_path = os.path.join(output_dir, image_filename)

                        image_data = base_image["image"]
                        image = Image.open(io.BytesIO(image_data))

                        image = correct_inverted_image(image)
                        image.save(image_path, format="PNG")

                        images.append(
                            {
                                "path": f"{pdf_name}_images/{image_filename}",
                                "name": image_filename,
                                "caption": caption,
                            }
                        )

        doc.close()
        return images
    except Exception as e:
        logging.error(f"Error extracting images: {e}")
        return []


# Routes
@app.route("/")
def index():
    """Serve the index page."""
    return render_template("index.html")


@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    """Handle PDF upload, generate XML, and extract images with captions."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)
            logging.info(f"File uploaded successfully: {filename}")

            # Extract text blocks and images
            text_blocks = extract_text_blocks(filepath)
            extracted_images = extract_images_with_captions(filepath)

            # Prepare prompt and call LLM
            api_key = "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"
            prompt = prepare_prompt(text_blocks, extracted_images)
            raw_xml_output = call_llm(prompt, api_key)

            if raw_xml_output:
                sanitized_xml = sanitize_xml_output(raw_xml_output)

                if sanitized_xml:
                    # Format the XML according to the new structure
                    formatted_xml = parse_and_format_xml(sanitized_xml)

                    if formatted_xml:
                        # Save the formatted XML to a file
                        xml_filename = os.path.splitext(filename)[0] + ".xml"
                        xml_filepath = os.path.join(
                            app.config["UPLOAD_FOLDER"], xml_filename
                        )
                        with open(xml_filepath, "w", encoding="utf-8") as xml_file:
                            xml_file.write(formatted_xml)

                        # Clean up the uploaded PDF file
                        os.remove(filepath)

                        return jsonify(
                            {
                                "message": "PDF processed and XML generated successfully",
                                "xml_file": xml_filename,
                                "xml_url": f"/xml/{xml_filename}",
                                "images": [
                                    {
                                        "name": img["name"],
                                        "caption": img["caption"],
                                        "url": f"/images/{img['path']}",
                                    }
                                    for img in extracted_images
                                ],
                            }
                        )

            return jsonify({"error": "Failed to process PDF"}), 500
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve extracted images."""
    return send_from_directory(app.config["EXTRACTED_IMAGES"], filename)


@app.route("/xml/<path:filename>")
def serve_xml(filename):
    """Serve generated XML files."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
