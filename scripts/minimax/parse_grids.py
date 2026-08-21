from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
import json
import re

def parse_init_para_freq(filepath, outpath):
    print(f"Parsing {filepath}...")
    
    grids = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # The file has a header from the API fetch, skip it until we hit '1_xk'
    start_idx = 0
    for i, line in enumerate(lines):
        if '1_xk' in line:
            start_idx = i
            break
            
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Match header like: 1_xk02_0.1100000E+01
        m = re.match(r'1_xk(\d+)_([\d\.E\+\-]+)', line)
        if m:
            n_points = int(m.group(1))
            energy_range = float(m.group(2))
            
            # The next 2 * n_points lines are the data
            # Typically weights first, then nodes (or vice versa)
            # Then 1 line for rms-error
            data = []
            while len(data) < 2 * n_points:
                i += 1
                if i >= len(lines): break
                l = lines[i].strip()
                if l:
                    data.append(float(l))
                
            i += 1
            rms_line = lines[i].strip()
            while not rms_line and i < len(lines) - 1:
                i += 1
                rms_line = lines[i].strip()
                
            rms_error = float(rms_line.split()[0])
            
            # Split into weights and nodes
            weights = data[:n_points]
            nodes = data[n_points:]
            
            if n_points not in grids:
                grids[n_points] = []
                
            grids[n_points].append({
                "energy_range": energy_range,
                "weights": weights,
                "nodes": nodes,
                "rms_error": rms_error
            })
            
        i += 1
        
    print(f"Parsed grids for N = {sorted(list(grids.keys()))}")
    
    # Save to JSON
    with open(outpath, 'w') as f:
        json.dump(grids, f, indent=2)
        
    print(f"Saved to {outpath}")

if __name__ == '__main__':
    in_file = str(_ROOT / 'init_para_freq.txt')
    out_file = str(_ROOT / 'minimax_freq_grids.json')
    parse_init_para_freq(in_file, out_file)
