import os
import csv

def parse_report(filepath):
    results = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return results
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('File Name') or line.startswith('-'):
            continue
            
        parts = line.split('|')
        if len(parts) >= 2:
            file_name = parts[0].strip()
            status = parts[1].strip()
            results[file_name] = status
            
    return results

def main():
    # Define the mapping of column headers to file names
    reports = [
        ("eval_llm_4", "evaluation_llm_report_4.txt"),
        ("eval_llm_8", "evaluation_llm_report_8.txt"),
        ("eval_llm_10", "evaluation_llm_report_10.txt"),
        ("eval_spataillm", "evaluation_spatiallm_report_29000.txt")
    ]
    
    # Load data
    all_data = {}
    all_files = set()
    
    for col_name, filename in reports:
        file_data = parse_report(filename)
        all_data[col_name] = file_data
        all_files.update(file_data.keys())
        
    # Sort files
    sorted_files = sorted(list(all_files))
    
    # Filter for failures
    failed_files = []
    for file_name in sorted_files:
        is_fail = False
        row_data = {}
        for col_name, _ in reports:
            status = all_data[col_name].get(file_name, "N/A")
            row_data[col_name] = status
            if status == "FAIL":
                is_fail = True
        
        if is_fail:
            failed_files.append((file_name, row_data))
            
    # Print table
    header = f"{'File name':<20} {'eval_llm_4':<12} {'eval_llm_8':<12} {'eval_llm_10':<12} {'eval_spataillm':<15}"
    print(header)
    
    for file_name, row_data in failed_files:
        line = f"{file_name:<20} {row_data['eval_llm_4']:<12} {row_data['eval_llm_8']:<12} {row_data['eval_llm_10']:<12} {row_data['eval_spataillm']:<15}"
        print(line)

    # Save to CSV
    csv_filename = "failures_report.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['File name', 'eval_llm_4', 'eval_llm_8', 'eval_llm_10', 'eval_spataillm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file_name, row_data in failed_files:
            row = {'File name': file_name}
            row.update(row_data)
            writer.writerow(row)
            
    print(f"\nResults saved to {csv_filename}")

if __name__ == "__main__":
    main()
