import re
import os
import xml.etree.ElementTree as ET

def process_metadata(line, parent_elem):
    """Extract metadata based on provided patterns."""
    metadata_patterns = {
        "title": r'(?i)^#+\s*(.*)',
        # Add additional patterns here for author, isbn, etc.
    }
    for key, pattern in metadata_patterns.items():
        match = re.match(pattern, line)
        if match:
            metadata_elem = ET.SubElement(parent_elem, key)
            metadata_elem.text = match.group(1)
            return True
    return False

def mmd_to_xml(markdown_dir, output_xml_file):
    root = ET.Element("books")

    for file in os.listdir(markdown_dir):
        if file.endswith('.mmd'):
            mmd_file_path = os.path.join(markdown_dir, file)
            with open(mmd_file_path, 'r', encoding='utf-8') as mmd_file:
                lines = mmd_file.readlines()

            book = ET.SubElement(root, "book", name=file.replace('.mmd', ''))
            metadata = ET.SubElement(book, "metadata")
            content = ET.SubElement(book, "content")
            paragraph_lines = []  # Initialize paragraph lines for each book

            in_metadata = True
            for line in lines:
                line = line.strip()
                if line.startswith('[MISSING_PAGE_EMPTY:'):
                    continue

                if in_metadata and not process_metadata(line, metadata):
                    in_metadata = False

                if not in_metadata and line:
                    if line.startswith('* '):  # Bullet points as separate paragraphs
                        paragraph = ET.SubElement(content, "paragraph")
                        paragraph.text = line[2:]  # Skip the bullet point marker
                    else:
                        # Aggregate lines into paragraphs
                        if not line or line == os.linesep:
                            if paragraph_lines:
                                paragraph = ET.SubElement(content, "paragraph")
                                paragraph.text = ' '.join(paragraph_lines)
                                paragraph_lines.clear()
                        else:
                            paragraph_lines.append(line)

            # Process any remaining lines as the last paragraph
            if paragraph_lines:
                paragraph = ET.SubElement(content, "paragraph")
                paragraph.text = ' '.join(paragraph_lines)

    tree = ET.ElementTree(root)
    tree.write(output_xml_file, encoding='utf-8', xml_declaration=True)

    return f'Combined XML file generated at: {output_xml_file}'

markdown_dir = 'C:/Users/elija/OneDrive/Desktop/ML/Markdown'
output_xml_file = 'C:/Users/elija/OneDrive/Desktop/ML/Markdown/all_books.xml'
result = mmd_to_xml(markdown_dir, output_xml_file)
print(result)
