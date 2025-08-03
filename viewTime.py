import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds since midnight."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}")

# Collect data
data = []
csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
print(f"Found {len(csv_files)} CSV files: {csv_files}")

for csv_file in csv_files:
    print(f"Processing file: {csv_file}")
    df = None
    for sep in ['\t', ',', None]:  # Try tab, comma, auto-detect
        try:
            df = pd.read_csv(csv_file, sep=sep)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            required_columns = {'startTime', 'endTime', 'fileName'}
            if len(df.columns) > 1 and required_columns.issubset(set(df.columns)):
                print(f"  - Successfully read with sep={sep}. Columns: {list(df.columns)}")
                break
            else:
                missing = required_columns - set(df.columns)
                print(f"  - Read with sep={sep} but missing required columns {missing} or too few columns ({len(df.columns)}). Trying next separator.")
        except Exception as e:
            print(f"  - Failed to read with sep={sep}: {e}")
            continue
    else:
        print(f"  - Skipping file: Unable to read with required columns.")
        continue

    # Now process the rows
    print(f"  - Required columns found. Processing {len(df)} rows.")
    for idx, row in df.iterrows():
        try:
            start_str = row['startTime']
            end_str = row['endTime']
            file_name = row['fileName']
            
            print(f"    - Row {idx}: startTime={start_str}, endTime={end_str}, fileName={file_name}")
            
            # Extract Unix timestamp from fileName
            match = re.search(r'(\d+)\.aedt', file_name)
            if not match:
                print(f"    - Skipping row {idx}: No timestamp found in fileName")
                continue
            start_ts = int(match.group(1))
            
            # Calculate elapsed
            start_sec = time_to_seconds(start_str)
            end_sec = time_to_seconds(end_str)
            if end_sec >= start_sec:
                elapsed_sec = end_sec - start_sec
            else:
                elapsed_sec = end_sec - start_sec + 86400  # Add 24 hours
            
            data.append({'start_ts': start_ts, 'elapsed_sec': elapsed_sec})
            print(f"    - Row {idx} processed successfully. Elapsed: {elapsed_sec} seconds")
        except Exception as e:
            print(f"    - Skipping row {idx} in {csv_file}: {e}")
            continue

# If no data, exit
if not data:
    print("No valid simulation data found in CSV files.")
else:
    print(f"Processed {len(data)} simulations.")
    # Compute relative x and y
    min_ts = min(d['start_ts'] for d in data)
    plot_data = [
        {
            'x': (d['start_ts'] - min_ts) / 3600.0,
            'y': d['elapsed_sec'] / 60.0
        }
        for d in data
    ]
    
    # Sort by x for better visualization
    plot_data.sort(key=lambda p: p['x'])
    
    x = [p['x'] for p in plot_data]
    y = [p['y'] for p in plot_data]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.xlabel('Hours since first simulation')
    plt.ylabel('Elapsed time (minutes)')
    plt.title('Simulation Speeds')
    plt.grid(True)
    plt.show()
