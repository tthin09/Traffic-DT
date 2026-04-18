import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

def analyze_vehicle_directions(rou_file):
    """
    Analyze vehicle counts from all 4 directions in SUMO route file.
    
    :param rou_file: Path to route.rou.xml file
    :return: Dictionary with detailed statistics
    """
    tree = ET.parse(rou_file)
    root = tree.getroot()
    
    # Initialize counters
    entry_counts = Counter()
    exit_counts = Counter()
    route_counts = Counter()
    type_counts = Counter()
    entry_type_counts = defaultdict(Counter)
    direction_details = defaultdict(list)
    
    # Process each vehicle
    for vehicle in root.findall('vehicle'):
        vehicle_id = vehicle.get('id')
        route_id = vehicle.get('route')
        vtype_id = vehicle.get('type')
        depart_time = float(vehicle.get('depart', 0))
        
        # Extract entry and exit from route name
        # Format: "route_<entry>_to_<exit>"
        if route_id and route_id.startswith('route_'):
            parts = route_id.replace('route_', '').split('_to_')
            if len(parts) == 2:
                entry = parts[0]
                exit_region = parts[1]
                
                # Extract vehicle type from vType ID
                # Format: "vType_<type>_<speed>"
                vehicle_type = "unknown"
                if vtype_id and '_' in vtype_id:
                    type_parts = vtype_id.split('_')
                    if len(type_parts) >= 2:
                        vehicle_type = type_parts[1]
                
                # Count entries
                entry_counts[entry] += 1
                exit_counts[exit_region] += 1
                route_counts[f"{entry}→{exit_region}"] += 1
                type_counts[vehicle_type] += 1
                entry_type_counts[entry][vehicle_type] += 1
                
                # Store details
                direction_details[entry].append({
                    'id': vehicle_id,
                    'type': vehicle_type,
                    'exit': exit_region,
                    'depart': depart_time
                })
    
    return {
        'entry_counts': entry_counts,
        'exit_counts': exit_counts,
        'route_counts': route_counts,
        'type_counts': type_counts,
        'entry_type_counts': entry_type_counts,
        'direction_details': direction_details
    }

def print_direction_summary(rou_file):
    """
    Print a formatted summary of vehicles from all directions.
    """
    stats = analyze_vehicle_directions(rou_file)
    
    entry_counts = stats['entry_counts']
    exit_counts = stats['exit_counts']
    type_counts = stats['type_counts']
    entry_type_counts = stats['entry_type_counts']
    route_counts = stats['route_counts']
    
    total_vehicles = sum(entry_counts.values())
    
    print("\n" + "=" * 60)
    print("VEHICLE DIRECTION ANALYSIS")
    print("=" * 60)
    print(f"Route File: {rou_file}")
    print(f"Total Vehicles: {total_vehicles}")
    print("=" * 60)
    
    # Print vehicles FROM each direction
    print("\n" + "─" * 60)
    print("🚗 VEHICLES FROM EACH DIRECTION (ENTRY)")
    print("─" * 60)
    
    directions = ['north', 'east', 'south', 'west']
    for direction in directions:
        count = entry_counts.get(direction, 0)
        percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
        
        # Get breakdown by vehicle type
        types_str = ""
        if direction in entry_type_counts:
            type_breakdown = []
            for vtype, vcount in sorted(entry_type_counts[direction].items()):
                type_breakdown.append(f"{vtype}:{vcount}")
            types_str = f" ({', '.join(type_breakdown)})"
        
        print(f"{direction.upper():10s}: {count:4d} vehicles ({percentage:5.1f}%){types_str}")
    
    # Print vehicles TO each direction
    print("\n" + "─" * 60)
    print("🚦 VEHICLES TO EACH DIRECTION (EXIT)")
    print("─" * 60)
    
    for direction in directions:
        count = exit_counts.get(direction, 0)
        percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"{direction.upper():10s}: {count:4d} vehicles ({percentage:5.1f}%)")
    
    # Print vehicle types summary
    print("\n" + "─" * 60)
    print("🚙 VEHICLE TYPES")
    print("─" * 60)
    
    for vtype in sorted(type_counts.keys()):
        count = type_counts[vtype]
        percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"{vtype.capitalize():15s}: {count:4d} vehicles ({percentage:5.1f}%)")
    
    # Print detailed breakdown by direction
    print("\n" + "─" * 60)
    print("📊 DETAILED BREAKDOWN BY ENTRY DIRECTION")
    print("─" * 60)
    
    for direction in directions:
        if direction in entry_type_counts:
            direction_total = sum(entry_type_counts[direction].values())
            print(f"\n{direction.upper()} ({direction_total} vehicles):")
            
            for vtype in sorted(entry_type_counts[direction].keys()):
                count = entry_type_counts[direction][vtype]
                percentage = (count / direction_total * 100) if direction_total > 0 else 0
                print(f"  {vtype.capitalize():12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Print top routes
    print("\n" + "─" * 60)
    print("🔀 TOP 10 ROUTES")
    print("─" * 60)
    
    for route, count in route_counts.most_common(10):
        percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"{route:25s}: {count:4d} vehicles ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    
    return stats

def save_direction_statistics_csv(rou_file, output_csv="direction_statistics.csv"):
    """
    Save direction statistics to CSV file.
    """
    import csv
    
    stats = analyze_vehicle_directions(rou_file)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write entry direction summary
        writer.writerow(['Entry Direction Summary'])
        writer.writerow(['Direction', 'Total', 'Motorcycles', 'Cars', 'Buses', 'Trucks'])
        
        total = sum(stats['entry_counts'].values())
        directions = ['north', 'east', 'south', 'west']
        
        for direction in directions:
            total_count = stats['entry_counts'].get(direction, 0)
            motorcycle_count = stats['entry_type_counts'][direction].get('motorcycle', 0)
            car_count = stats['entry_type_counts'][direction].get('car', 0)
            bus_count = stats['entry_type_counts'][direction].get('bus', 0)
            truck_count = stats['entry_type_counts'][direction].get('truck', 0)
            
            writer.writerow([
                direction.capitalize(),
                total_count,
                motorcycle_count,
                car_count,
                bus_count,
                truck_count
            ])
        
        writer.writerow([])
        
        # Write route statistics
        writer.writerow(['Route Statistics'])
        writer.writerow(['Route', 'Count', 'Percentage'])
        
        for route, count in sorted(stats['route_counts'].items()):
            percentage = (count / total * 100) if total > 0 else 0
            writer.writerow([route, count, f"{percentage:.1f}%"])
    
    print(f"✅ Statistics saved to {output_csv}")

def get_direction_summary(rou_file):
    """
    Get a quick summary dictionary for programmatic use.
    
    :return: Dictionary with counts from each direction
    """
    stats = analyze_vehicle_directions(rou_file)
    
    summary = {
        'north': stats['entry_counts'].get('north', 0),
        'east': stats['entry_counts'].get('east', 0),
        'south': stats['entry_counts'].get('south', 0),
        'west': stats['entry_counts'].get('west', 0),
        'total': sum(stats['entry_counts'].values())
    }
    
    return summary

if __name__ == "__main__":
    import sys
    
    # Default route_name
    rou_name = "tphcm-1h"
    
    # Allow command-line argument
    if len(sys.argv) > 1:
        rou_name = sys.argv[1]
        
    rou_file = f"sumo_files/{rou_name}/route.rou.xml"
    
    # Print detailed analysis
    stats = print_direction_summary(rou_file)
    
    # Save to CSV
    save_direction_statistics_csv(rou_file, "direction_statistics.csv")
    
    # Quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    summary = get_direction_summary(rou_file)
    print(f"North: {summary['north']} vehicles")
    print(f"East:  {summary['east']} vehicles")
    print(f"South: {summary['south']} vehicles")
    print(f"West:  {summary['west']} vehicles")
    print(f"TOTAL: {summary['total']} vehicles")
    print("=" * 60 + "\n")