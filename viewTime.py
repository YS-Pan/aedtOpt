import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime
import math
import numpy as np
import hashlib
import colorsys

def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds since midnight."""
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}")

def get_nice_step(max_val, target_ticks=8):
    if max_val == 0:
        return 1
    step = max_val / (target_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(step))
    normalized = step / magnitude
    if normalized < 1.5:
        return 1 * magnitude
    elif normalized < 3:
        return 2 * magnitude
    elif normalized < 7:
        return 5 * magnitude
    else:
        return 10 * magnitude

def gaussian_kernel_smoother(x_data, y_data, fine_x, sigma):
    """Compute Gaussian kernel smoothed values on fine_x."""
    smoothed = np.zeros_like(fine_x)
    for i, fx in enumerate(fine_x):
        weights = np.exp(- (x_data - fx)**2 / (2 * sigma**2))
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * y_data) / np.sum(weights)
    return smoothed

def get_color_from_filename(file_name):
    last_10 = file_name[-10:]
    seed = int(hashlib.sha256(last_10.encode()).hexdigest(), 16)
    hue = (seed % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
    return (r, g, b)

# Collect data
data = []
csv_file = 'output.csv'
if os.path.exists(csv_file):
    df = None
    for sep in ['\t', ',', None]:  # Try tab, comma, auto-detect
        try:
            df = pd.read_csv(csv_file, sep=sep)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            required_columns = {'startTime', 'endTime', 'fileName'}
            if len(df.columns) > 1 and required_columns.issubset(set(df.columns)):
                break
            else:
                continue
        except Exception:
            continue
    if df is not None:
        # Now process the rows
        for idx, row in df.iterrows():
            if pd.isna(row['startTime']) or pd.isna(row['endTime']) or pd.isna(row['fileName']):
                continue  # Skip incomplete rows
            
            start_str = str(row['startTime'])
            end_str = str(row['endTime'])
            file_name = str(row['fileName'])
            
            valid = True
            start_ts = None
            elapsed_sec = 0
            
            # Try to extract Unix timestamp from fileName
            try:
                match = re.search(r'(\d+)\.aedt', file_name)
                if not match:
                    raise ValueError("No timestamp in fileName")
                start_ts = int(match.group(1))
            except:
                valid = False
            
            # Try to calculate elapsed
            try:
                start_sec = time_to_seconds(start_str)
                end_sec = time_to_seconds(end_str)
                if end_sec >= start_sec:
                    elapsed_sec = end_sec - start_sec
                else:
                    elapsed_sec = end_sec - start_sec + 86400  # Add 24 hours
            except:
                valid = False
            
            data.append({'idx': idx, 'start_ts': start_ts, 'elapsed_sec': elapsed_sec, 'start_str': start_str if valid else None, 'valid': valid, 'file_name': file_name})
else:
    print(f"{csv_file} not found.")

# If no data, exit
if not data:
    print("No valid simulation data found in CSV files.")
else:
    # Sort data by original index first to group them logically
    data.sort(key=lambda d: d['idx'])
    
    # Assign provisional start_ts for points without them by interpolating locally
    for i in range(len(data)):
        if data[i]['start_ts'] is None:
            # Find nearest left valid ts
            left_ts = None
            for j in range(i-1, -1, -1):
                if data[j]['start_ts'] is not None:
                    left_ts = data[j]['start_ts']
                    break
            # Find nearest right valid ts
            right_ts = None
            for j in range(i+1, len(data)):
                if data[j]['start_ts'] is not None:
                    right_ts = data[j]['start_ts']
                    break
            if left_ts is not None and right_ts is not None:
                # Interpolate
                num_between = (i - (i-1)) + ((j if 'j' in locals() else len(data)) - i)  # Approximate points between
                data[i]['start_ts'] = left_ts + (right_ts - left_ts) / 2  # Middle for simplicity
            elif left_ts is not None:
                data[i]['start_ts'] = left_ts + 60  # Default 1 min after left
            elif right_ts is not None:
                data[i]['start_ts'] = right_ts - 60  # Default 1 min before right
            else:
                data[i]['start_ts'] = i * 60  # Fallback if no valid ts at all
    
    # Now sort by start_ts for plotting
    data.sort(key=lambda d: d['start_ts'])
    
    # Compute x
    min_ts = min(d['start_ts'] for d in data if d['start_ts'] is not None)
    for d in data:
        d['x'] = (d['start_ts'] - min_ts) / 3600.0 if d['start_ts'] is not None else 0
    
    # Adjust x for groups of invalid (even distribution between valid)
    i = 0
    while i < len(data):
        if data[i]['valid']:
            i += 1
            continue
        start_group = i
        while i < len(data) and not data[i]['valid']:
            i += 1
        end_group = i
        k = end_group - start_group
        if start_group == 0 or end_group == len(data):
            continue
        left_x = data[start_group - 1]['x']
        right_x = data[end_group]['x'] if end_group < len(data) else data[-1]['x']
        delta = right_x - left_x
        step = delta / (k + 1)
        for j in range(k):
            data[start_group + j]['x'] = left_x + (j + 1) * step
    
    # Prepare plot data
    plot_data = sorted(data, key=lambda d: d['x'])
    x = np.array([d['x'] for d in plot_data])
    y = np.array([d['elapsed_sec'] / 60.0 for d in plot_data])
    min_x = min(x) if len(x) > 0 else 0
    max_x = max(x) if len(x) > 0 else 0
    
    # Compute start_hour_frac
    valid_data = [d for d in data if d['valid'] and d['start_str'] is not None]
    if valid_data:
        min_valid_ts = min(d['start_ts'] for d in valid_data)
        min_valid_d = next(d for d in valid_data if d['start_ts'] == min_valid_ts)
        rel_hours_to_min_valid = (min_valid_ts - min_ts) / 3600.0
        min_valid_frac = time_to_seconds(min_valid_d['start_str']) / 3600.0
        start_hour_frac = (min_valid_frac - rel_hours_to_min_valid) % 24
    else:
        start_hour_frac = 0
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter points separately
    valid_idx = np.array([d['valid'] for d in plot_data])
    ax.scatter(x[valid_idx], y[valid_idx], color='red', alpha=0.5)
    
    # Invalid points with custom colors
    for i in np.where(~valid_idx)[0]:
        d = plot_data[i]
        color = get_color_from_filename(d['file_name'])
        ax.scatter(x[i], y[i], color=color, alpha=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Elapsed time (minutes)')
    ax.set_title('Simulation Speeds')
    ax.grid(True, color='gainsboro', linestyle='-', linewidth=0.5, alpha=0.6)
    
    # Fine x for smoothing, limited to data range
    if len(x) > 0:
        fine_x = np.linspace(min_x, max_x, 2000)
        
        # Compute sigma for Gaussian kernel
        total_points = len(data)
        window_points = max(1, int(0.05 * total_points))
        avg_spacing = (max_x - min_x) / (total_points - 1) if total_points > 1 else 1
        sigma = (window_points * avg_spacing) / 3.0  # Approximate sigma for Gaussian window
        
        # Local average for elapsed time (only valid points)
        valid_x = x[valid_idx]
        valid_y = y[valid_idx]
        if len(valid_y) > 0:
            global_avg = np.mean(valid_y)
            local_avg = gaussian_kernel_smoother(valid_x, valid_y, fine_x, sigma)
            ax.plot(fine_x, local_avg, color='orange', label=f'avg. time (global: {global_avg:.2f} min)', linewidth=2)
        
        # Local failure rate
        failure_y = np.array([0 if valid else 1 for valid in valid_idx])
        global_failure = (np.sum(failure_y) / len(failure_y) * 100) if len(failure_y) > 0 else 0
        local_failure = gaussian_kernel_smoother(x, failure_y, fine_x, sigma) * 100
        
        ax2 = ax.twinx()
        ax2.set_ylabel('Failure rate (%)')
        ax2.set_ylim(0, 100)
        ax2.plot(fine_x, local_failure, color='darkblue', label=f'avg. failure rate (global: {global_failure:.2f} %)', linewidth=2, alpha=0.5)
    
    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels() if 'ax2' in locals() else ([], [])
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # Custom x-axis ticks with left extension
    left_offset = start_hour_frac % 1
    left_x = -left_offset
    range_for_step = max_x - left_x
    step = get_nice_step(range_for_step)
    hour_ticks = np.arange(0, range_for_step + step, step) + left_x
    
    hour_labels = []
    for tick in hour_ticks:
        total_hours = start_hour_frac + tick
        label_hour = total_hours % 24
        hour_labels.append(f"{int(label_hour):02d}")
    
    # Trim last tick if it creates large blank at end
    if len(hour_ticks) > 0 and hour_ticks[-1] - max_x > step / 2:
        hour_ticks = hour_ticks[:-1]
        hour_labels = hour_labels[:-1]
    
    # Add small padding to right limit
    right_x = max_x + (step * 0.1)  # Small padding to avoid cutting lines
    
    # Day change ticks and lines
    hours_to_first_midnight = (24 - (start_hour_frac % 24)) % 24
    if hours_to_first_midnight == 0:
        hours_to_first_midnight = 24
    day_ticks = []
    current_rel = hours_to_first_midnight
    day = 1
    while current_rel <= max_x + 1:
        day_ticks.append(current_rel)
        ax.axvline(current_rel, color='gray', linestyle='--', alpha=0.7)
        current_rel += 24
        day += 1
    day_labels = [f"{d}d" for d in range(1, len(day_ticks) + 1)]
    
    # Combine ticks and labels
    all_ticks = sorted(set(list(hour_ticks) + day_ticks))
    all_labels = []
    for tick in all_ticks:
        label_parts = []
        h_idx = np.where(np.isclose(hour_ticks, tick))[0]
        if len(h_idx) > 0:
            label_parts.append(hour_labels[h_idx[0]])
        for d_idx, d_tick in enumerate(day_ticks):
            if abs(tick - d_tick) < 1e-6:
                label_parts.append(day_labels[d_idx])
                break
        all_labels.append('\n'.join(label_parts))
    
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels)
    
    # Set x-limits
    ax.set_xlim(left=left_x, right=right_x)
    
    # Set y-limits starting from 0
    if len(y) > 0:
        ax.set_ylim(0, max(y) * 1.1)
    
    # Export figure
    now = datetime.datetime.now()
    fig_name = now.strftime("time_%Y%m%d_%H%M%S.png")
    plt.savefig(fig_name)