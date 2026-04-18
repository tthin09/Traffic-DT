import xml.etree.ElementTree as ET

def modify_vehicle_types(input_file, output_file, new_accel=2.5, new_max_speed=12.0):
    """
    Modify all vType elements to set new accel and maxSpeed values.
    
    Args:
        input_file: Path to input route.rou.xml file
        output_file: Path to save modified file
        new_accel: New acceleration value (default: 2.5)
        new_max_speed: New maximum speed value (default: 12.0)
    """
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Counter for modified vTypes
    count = 0
    
    # Find all vType elements and modify them
    for vtype in root.findall('vType'):
        vtype.set('accel', str(new_accel))
        vtype.set('maxSpeed', str(new_max_speed))
        count += 1
    
    # Write the modified XML to output file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Modified {count} vehicle types")
    print(f"New acceleration: {new_accel} m/s²")
    print(f"New max speed: {new_max_speed} m/s")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    input_path = r"c:\Users\TPC\Desktop\Code\HK251\DACN\TrafficDT\sumo_files\tphcm-1h\route.rou.xml"
    output_path = r"c:\Users\TPC\Desktop\Code\HK251\DACN\TrafficDT\sumo_files\tphcm-1h\route_modified.rou.xml"
    
    modify_vehicle_types(input_path, output_path, new_accel=6, new_max_speed=30.0)