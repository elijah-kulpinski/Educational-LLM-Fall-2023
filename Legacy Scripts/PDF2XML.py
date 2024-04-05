import os
import PyPDF2
import xml.etree.ElementTree as ET
import re
import nltk

# Ensure NLTK is installed and the punkt tokenizer models are downloaded
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ''
    print(f"Text successfully extracted from {os.path.basename(pdf_path)}")

    # Remove non-printable characters except for whitespace characters
    cleaned_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    return cleaned_text

def extract_metadata(text):
    # Generalized regular expressions for extracting metadata
    title_pattern = r'(?i)title[:\s]*([\w\s,]+)'
    subtitle_pattern = r'(?i)subtitle[:\s]*([\w\s,]+)'
    author_pattern = r'(?i)author[s]?[:\s]*([\w\s,]+)'
    isbn_pattern = r'(?i)ISBN[-\s]*:?[\s]*([\d\-]+)'
    edition_pattern = r'(?i)edition[:\s]*([\w\s,]+)'
    publisher_pattern = r'(?i)publisher[:\s]*([\w\s,]+)'
    publication_year_pattern = r'(?i)publication\s+year[:\s]*([\d]{4})'
    chapter_pattern = r'Chapter\s*(\d+):\s*(.*)' 
    section_heading_pattern = r'Section:\s*(.*)'
    page_number_pattern = r'Page\s*(\d+)'
    references_pattern = r'References:\s*(.*)'
    index_terms_pattern = r'Index Terms:\s*(.*)'
    figures_pattern = r'Figure\s*(\d+):\s*(.*)'
    footnotes_pattern = r'Footnote:\s*(.*)'
    table_of_contents_pattern = r'(?:Contents|Table\s+of\s+Contents|Index)\s+(?:.\n)?\s(?:Chapter\s+1|Introduction)'

    title = re.search(title_pattern, text)
    subtitle = re.search(subtitle_pattern, text)
    author = re.search(author_pattern, text)
    isbn = re.search(isbn_pattern, text)
    edition = re.search(edition_pattern, text)
    publisher = re.search(publisher_pattern, text)
    publication = re.search(publication_year_pattern, text)
    chapter = re.search(chapter_pattern, text)
    section = re.search(section_heading_pattern, text)
    page_number = re.search(page_number_pattern, text)
    references = re.search(references_pattern, text)
    index_terms = re.search(index_terms_pattern, text)
    figures = re.search(figures_pattern, text)
    footnotes = re.search(footnotes_pattern, text)
    toc = re.search(table_of_contents_pattern, text)

    metadata = {
        'title': title.group(1) if title else 'Unknown',
        'subtitle': subtitle.group(1) if subtitle else 'Unknown',
        'author': author.group(1) if author else 'Unknown',
        'isbn': isbn.group(1) if isbn else 'Unknown',
        'edition': edition.group(1) if edition else 'Unknown',
        'publisher': publisher.group(1) if publisher else 'Unknown',
        'publication': publication.group(1) if publication else 'Unknown',
        'chapter': chapter.group(1) if chapter else 'Unknown',
        'section': section.group(1) if section else 'Unknown',
        'page_number': page_number.group(1) if page_number else 'Unknown',
        'references': references.group(1) if references else 'Unknown',
        'index_terms': index_terms.group(1) if index_terms else 'Unknown',
        'figures': figures.group(1) if figures else 'Unknown',
        'footnotes': footnotes.group(1) if footnotes else 'Unknown',
        'table_of_contents': toc.group(1) if toc else 'Unknown'
    }
    return metadata

def extract_paragraphs(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    paragraphs = []
    paragraph = ""
    for sentence in sentences:
        if paragraph and re.match(r"^[A-Z]", sentence):
            paragraphs.append(paragraph.strip())
            paragraph = sentence
        else:
            paragraph += " " + sentence
    if paragraph:
        paragraphs.append(paragraph.strip())
    return paragraphs

def process_pdf_files(pdf_paths):
    root = ET.Element("textbook_dataset")

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        metadata = extract_metadata(text)
        paragraphs = extract_paragraphs(text)

        book_element = ET.SubElement(root, "book")
        metadata_element = ET.SubElement(book_element, "metadata")
        for key, value in metadata.items():
            ET.SubElement(metadata_element, key).text = value

        for paragraph in paragraphs:
            paragraph_element = ET.SubElement(book_element, "paragraph")
            paragraph_element.text = paragraph

    return root

def write_xml_to_file(root_element, output_file_path):
    tree = ET.ElementTree(root_element)
    with open(output_file_path, 'wb') as xml_file:
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    folder_path = "C:/Users/elija/OneDrive/Desktop/ML/ML-Textbook-Dataset/"
    pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    combined_xml = process_pdf_files(pdf_paths)
    output_file_path = os.path.join(folder_path, "combined_dataset.xml")
    write_xml_to_file(combined_xml, output_file_path)