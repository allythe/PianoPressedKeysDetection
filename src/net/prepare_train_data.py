import xml.etree.ElementTree as ET

# Read XML content from a file
file_path = 'annotations.xml'  # Replace with the path to your XML file
with open(file_path, 'r', encoding='utf-8') as file:
    xml_content = file.read()

# Parse the XML content
tree = ET.ElementTree(ET.fromstring(xml_content))
root = tree.getroot()

# Extract image names and 'Pressed' states
results = []
for image in root.findall(".//image"):
    name = image.get("name")
    pressed_state = image.find(".//attribute[@name='Pressed']").text
    results.append((name, pressed_state))

# Print the extracted results
for name, pressed in results:
    print(f"Image: {name}, Pressed: {pressed}")