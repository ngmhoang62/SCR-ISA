import os
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List

def extract_aspect_terms_to_csv(xml_path: str, csv_path: str) -> int:
    """Parses an XML file and extracts aspect term details to a CSV.
    
    Skips any sentence containing a conflict polarity aspect term.
    
    Args:
        xml_path: Path to the input test XML dataset.
        csv_path: Path to write the output CSV data.
    Returns:
        Number of aspect term rows written.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Input file not found: {xml_path}")
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sentences = root.findall(".//sentence")
    
    csv_rows = []
    
    for sentence in sentences:
        text_elem = sentence.find("text")
        text = text_elem.text.strip() if (text_elem is not None and text_elem.text) else ""
        
        aspect_terms = sentence.findall(".//aspectTerm")
        if not aspect_terms:
            continue

        # Gather other aspect terms
        for term in aspect_terms:
            polarity = term.attrib.get("polarity", "").lower()
            if polarity in ["positive", "negative", "neutral"]:
                term_text = term.attrib.get("term", "").strip()
                is_implicit = term.attrib.get("implicit_sentiment", "").strip()
                
                csv_rows.append({
                    "text": text,
                    "aspect_term": term_text,
                    "polarity": polarity,
                    "is_implicit": is_implicit
                })
                
    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Save fields to CSV file
    headers = ["text", "aspect_term", "polarity", "is_implicit"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    return len(csv_rows)

def main():
    # Define XML inputs and CSV outputs
    laptops_xml = os.path.join("data", "inputs", "laptops", "Laptops_Test_Gold_Implicit_Labeled.xml")
    laptops_csv = os.path.join("data", "inputs", "laptops", "laptops_test_extracted.csv")
    
    restaurants_xml = os.path.join("data", "inputs", "restaurants", "Restaurants_Test_Gold_Implicit_Labeled.xml")
    restaurants_csv = os.path.join("data", "inputs", "restaurants", "restaurants_test_extracted.csv")
    
    print("Running XML test dataset CSV extraction...")
    
    try:
        laptops_count = extract_aspect_terms_to_csv(laptops_xml, laptops_csv)
        print(f"Success: Extracted Laptop Test dataset. Written {laptops_count} samples to '{laptops_csv}'.")
    except Exception as e:
        print(f"Error extracting Laptops Test dataset: {e}")
        
    try:
        restaurants_count = extract_aspect_terms_to_csv(restaurants_xml, restaurants_csv)
        print(f"Success: Extracted Restaurant Test dataset. Written {restaurants_count} samples to '{restaurants_csv}'.")
    except Exception as e:
        print(f"Error extracting Restaurants Test dataset: {e}")

if __name__ == "__main__":
    main()
