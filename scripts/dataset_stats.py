import os
import xml.etree.ElementTree as ET
from typing import Dict

def parse_xml_stats(file_path: str) -> Dict[str, int]:
    """Parses a dataset XML file and counts aspect terms and sentiment properties."""
    stats = {
        "Total": 0,
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0,
        "Conflict": 0,
        "Implicit": 0
    }
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset XML file not found: {file_path}")
        
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Locate all aspectTerm tags recursively
    aspect_terms = root.findall(".//aspectTerm")
    
    for term in aspect_terms:
        stats["Total"] += 1
        
        # Count polarity types
        polarity = term.attrib.get("polarity", "").lower()
        if polarity == "positive":
            stats["Positive"] += 1
        elif polarity == "negative":
            stats["Negative"] += 1
        elif polarity == "neutral":
            stats["Neutral"] += 1
        elif polarity == "conflict":
            stats["Conflict"] += 1
            
        # Count implicit sentiment labels where implicit_sentiment is True
        implicit = term.attrib.get("implicit_sentiment", "").lower()
        if implicit == "true":
            stats["Implicit"] += 1
            
    return stats

def parse_csv_stats(file_path: str) -> Dict[str, int]:
    """Parses a dataset CSV file and counts polarity properties."""
    stats = {
        "Total": 0,
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0,
        "Conflict": 0,
        "Implicit": 0
    }
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset CSV file not found: {file_path}")
        
    import csv
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["Total"] += 1
            polarity = row.get("polarity", "").strip().lower()
            if polarity == "positive":
                stats["Positive"] += 1
            elif polarity == "negative":
                stats["Negative"] += 1
            elif polarity == "neutral":
                stats["Neutral"] += 1
            elif polarity == "conflict":
                stats["Conflict"] += 1
                
    return stats

def aggregate_stats(train_stats: Dict[str, int], test_stats: Dict[str, int]) -> Dict[str, int]:
    """Sums count dictionaries from different splits to produce combined metrics."""
    return {key: train_stats[key] + test_stats[key] for key in train_stats}

def format_split_report(split_name: str, stats: Dict[str, int]) -> str:
    """Formats split statistics dictionary into a clean line-by-line format."""
    return (
        f"{split_name}:\n"
        f"- Total: {stats['Total']}\n"
        f"- Positive: {stats['Positive']}\n"
        f"- Negative: {stats['Negative']}\n"
        f"- Neutral: {stats['Neutral']}\n"
        f"- Conflict: {stats['Conflict']}\n"
        f"- Implicit: {stats['Implicit']}"
    )

def main():
    # Set relative paths to data files
    laptops_dir = os.path.join("data", "inputs", "laptops")
    laptops_train = os.path.join(laptops_dir, "Laptops_Train_v2_Implicit_Labeled.xml")
    laptops_test = os.path.join(laptops_dir, "Laptops_Test_Gold_Implicit_Labeled.xml")

    restaurants_dir = os.path.join("data", "inputs", "restaurants")
    restaurants_train = os.path.join(restaurants_dir, "Restaurants_Train_v2_Implicit_Labeled.xml")
    restaurants_test = os.path.join(restaurants_dir, "Restaurants_Test_Gold_Implicit_Labeled.xml")

    vietnamese_dir = os.path.join("data", "inputs", "vietnamese")
    vietnamese_test = os.path.join(vietnamese_dir, "technologies_implicit_test.csv")

    # Compute statistics for laptops
    laptop_train_stats = parse_xml_stats(laptops_train)
    laptop_test_stats = parse_xml_stats(laptops_test)
    laptop_combined_stats = aggregate_stats(laptop_train_stats, laptop_test_stats)

    # Compute statistics for restaurants
    restaurant_train_stats = parse_xml_stats(restaurants_train)
    restaurant_test_stats = parse_xml_stats(restaurants_test)
    restaurant_combined_stats = aggregate_stats(restaurant_train_stats, restaurant_test_stats)

    # Compute statistics for Vietnamese Technology
    vietnamese_stats = parse_csv_stats(vietnamese_test)

    # Build the report content
    report_lines = [
        "=" * 80,
        format_split_report("Laptop - Train", laptop_train_stats),
        "",
        format_split_report("Laptop - Test", laptop_test_stats),
        "",
        format_split_report("Laptop - Combined", laptop_combined_stats),
        "=" * 80,
        format_split_report("Restaurant - Train", restaurant_train_stats),
        "",
        format_split_report("Restaurant - Test", restaurant_test_stats),
        "",
        format_split_report("Restaurant - Combined", restaurant_combined_stats),
        "=" * 80,
        format_split_report("Vietnamese Technology - Test", vietnamese_stats),
        "=" * 80
    ]
    report_content = "\n".join(report_lines) + "\n"

    # Print output to stdout
    print(report_content)
    
    # Save the output file
    output_path = os.path.join("data", "dataset_stats.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Dataset statistics report saved to: {output_path}")

if __name__ == "__main__":
    main()
