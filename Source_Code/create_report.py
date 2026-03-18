from simulation import AdaptiveQoSDashboard
import os
import sys
import pandas as pd

# Ensure results dir exists
results_dir = "results"
summary_file = os.path.join(results_dir, "simulation_summary.csv")

if not os.path.exists(summary_file):
    print("Error: summary file not found")
    sys.exit(1)

print("Patching summary data...")
try:
    df = pd.read_csv(summary_file)
    if 'QoS Score' not in df.columns:
        # Calculate a proxy QoS Score: Reliability * 100
        # If Reliability is missing, use 0
        if 'Reliability' in df.columns:
            df['QoS Score'] = df['Reliability'] * 100
        else:
             df['QoS Score'] = 0
        df.to_csv(summary_file, index=False)
        print("Added 'QoS Score' column.")
    else:
        print("'QoS Score' already exists.")

    print("Generating report...")
    dashboard = AdaptiveQoSDashboard()
    # Generate HTML report
    dashboard.generate_report("qos_report.html")
    # Export CSV summary (will overwrite the patched one, but effectively same content)
    dashboard.export_csv_summary("qos_summary_export.csv")
    print("Report generation complete.")
except Exception as e:
    print(f"Error generating report: {e}")
    # Print stack trace
    import traceback
    traceback.print_exc()
    sys.exit(1)
