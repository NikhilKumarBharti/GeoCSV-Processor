import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob2 as glob
import os
from obspy import UTCDateTime
from obspy.core import Stream, Trace, Stats
from scipy.io import wavfile
import scipy.signal as signal

def read_geocsv(file_path):
    """Read a GeoCSV file and convert to ObsPy Stream"""
    # Read the header to get metadata
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                parts = line.strip('# \n').split(': ', 1)
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]
            else:
                break
    
    # Read the actual data
    data = pd.read_csv(file_path, comment='#')
    
    # Check column names and get the data column
    columns = data.columns.tolist()
    if len(columns) < 2:
        raise ValueError(f"CSV file {file_path} doesn't have enough columns")
    
    # Assuming the second column contains the sample data
    data_column = columns[1]
    
    # Extract key metadata
    sid = metadata.get('SID', 'unknown')
    
    # Parse file name to get components
    filename = os.path.basename(file_path)
    parts = filename.split('.')
    
    # Extract components from filename
    if len(parts) >= 5:
        network = parts[0]
        station = parts[1]
        location = parts[2]  # This will be 00 or 10
        channel = parts[3]   # This will be BH1, BH2, or BHZ
    else:
        # Fallback to SID parsing
        sid_parts = sid.split('_')
        if len(sid_parts) >= 3:
            network = sid_parts[0]
            station = sid_parts[1]
            # Try to extract location and channel from the filename
            channel = sid_parts[-1]
            location = sid_parts[2] if len(sid_parts) > 3 else ''
        else:
            network = ''
            station = sid
            channel = ''
            location = ''
    
    sample_rate = float(metadata.get('sample_rate_hz', 1.0))
    start_time = UTCDateTime(metadata.get('start_time', '2000-01-01T00:00:00Z'))
    
    # Create ObsPy Stats object
    stats = Stats({
        'network': network,
        'station': station,
        'location': location,
        'channel': channel,
        'starttime': start_time,
        'sampling_rate': sample_rate,
        'npts': len(data),
        'filename': filename
    })
    
    # Create ObsPy Trace with the sample data
    tr = Trace(data=np.array(data[data_column]), header=stats)
    
    return Stream(traces=[tr])

def counts_to_acceleration(stream, sensitivity=1.0):
    """Convert counts to acceleration (m/s^2)"""
    for tr in stream:
        # Apply instrument correction (simplified)
        tr.data = tr.data / sensitivity
        tr.stats.units = 'm/s^2'
    
    return stream

def process_directory(directory):
    """Process all CSV files in a directory, keeping location codes separate"""
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    # Organize by location code
    location_streams = {}
    
    for file in csv_files:
        print(f"Processing {file}")
        stream = read_geocsv(file)
        
        # Apply conversion to acceleration
        accel_stream = counts_to_acceleration(stream, sensitivity=2800.0)
        
        # Group by location code
        location = accel_stream[0].stats.location
        if location not in location_streams:
            location_streams[location] = []
        
        location_streams[location].append(accel_stream)
    
    return location_streams

def calculate_pga_pgv(stream):
    """Calculate Peak Ground Acceleration (PGA) and Peak Ground Velocity (PGV)"""
    for tr in stream:
        # Get acceleration data
        acceleration = tr.data
        
        # Calculate PGA (absolute maximum acceleration)
        pga = np.max(np.abs(acceleration))
        
        # Calculate velocity by integrating acceleration
        # First, detrend to remove any DC offset
        detrended_acc = signal.detrend(acceleration)
        
        # Integrate to get velocity (using cumulative trapezoidal integration)
        dt = 1.0 / tr.stats.sampling_rate
        velocity = np.cumsum(detrended_acc) * dt
        
        # Apply a highpass filter to remove drift in velocity
        if len(velocity) > 10:  # Only apply if enough data points
            sos = signal.butter(4, 0.075, 'highpass', fs=tr.stats.sampling_rate, output='sos')
            velocity = signal.sosfilt(sos, velocity)
        
        # Calculate PGV (absolute maximum velocity)
        pgv = np.max(np.abs(velocity))
        
        # Store PGA and PGV in trace stats
        tr.stats.pga = pga
        tr.stats.pgv = pgv
        
        # Store velocity data for potential use
        tr.stats.velocity = velocity
    
    return stream

def classify_by_frequency_content(stream):
    """Classify signal based on PGA/PGV ratio"""
    for tr in stream:
        # Ensure PGA and PGV have been calculated
        if not hasattr(tr.stats, 'pga') or not hasattr(tr.stats, 'pgv'):
            raise ValueError("PGA/PGV not calculated. Run calculate_pga_pgv first.")
        
        # Avoid division by zero
        if tr.stats.pgv > 0:
            ratio = tr.stats.pga / tr.stats.pgv
        else:
            ratio = float('inf')
        
        # Classify based on ratio
        if ratio > 1.2:
            classification = "High-frequency content (PGA/PGV > 1.2)"
        elif 0.8 <= ratio <= 1.2:
            classification = "Intermediate-frequency content (0.8 ≤ PGA/PGV ≤ 1.2)"
        else:  # ratio < 0.8
            classification = "Low-frequency content (PGA/PGV < 0.8)"
        
        # Store classification and ratio
        tr.stats.frequency_class = classification
        tr.stats.pga_pgv_ratio = ratio
    
    return stream

def create_accelerogram(stream, output_dir="accelerograms"):
    """Create and save accelerogram plot with classification info"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for tr in stream:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot acceleration data
        times = tr.times()
        ax.plot(times, tr.data, 'b-', linewidth=1.5)
        
        # Add labels and title
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
        
        # Create a detailed title with classification information
        title = (f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}\n"
                 f"PGA: {tr.stats.pga:.6f} m/s², PGV: {tr.stats.pgv:.6f} m/s, Ratio: {tr.stats.pga_pgv_ratio:.3f}\n"
                 f"Classification: {tr.stats.frequency_class}")
        ax.set_title(title, fontsize=10)
        
        # Add a text box with classification
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax.text(0.05, 0.95, tr.stats.frequency_class, transform=ax.transAxes,
        #         verticalalignment='top', bbox=props, fontsize=12)
        
        # Add grid and improve appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        
        # Create a clean filename
        filename = f"{output_dir}/{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}_accelerogram.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        
        print(f"Created accelerogram: {filename}")

# def create_sonification(stream, output_dir="sonification"):
#     """Create WAV files from seismic data"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     for tr in stream:
#         # Get data
#         data = tr.data.copy()
        
#         # Normalize the data to [-1, 1] range
#         if np.max(np.abs(data)) > 0:
#             data = data / np.max(np.abs(data))
        
#         # Target sample rate for audio
#         target_sample_rate = 22050  # Common audio rate
        
#         # Resample to the target rate
#         samples_needed = int(len(data) * (target_sample_rate / tr.stats.sampling_rate))
#         data = signal.resample(data, samples_needed)
        
#         # Apply frequency shift to make the signal more audible
#         # (Optional) Time compression - speeds up playback
#         time_compression = 10  # Play 10x faster
#         data = signal.resample(data, len(data) // time_compression)
        
#         # Save as WAV file
#         filename = f"{output_dir}/{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}.wav"
#         wavfile.write(filename, target_sample_rate, data.astype(np.float32))
#         print(f"Created sonification: {filename}")

def create_sonification(stream, output_dir="sonification"):

    volume_factor=100.0
    """Create WAV files from seismic data with minimal filtering to preserve original characteristics"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for tr in stream:
        # Get data
        data = tr.data.copy()
        
        # Normalize the data to [-1, 1] range
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        
        # Target sample rate for audio
        target_sample_rate = 22050  # Common audio rate
        
        # Resample to the target rate
        samples_needed = int(len(data) * (target_sample_rate / tr.stats.sampling_rate))
        data = signal.resample(data, samples_needed)
        
        # Preserve more noise by avoiding heavy filtering
        # Just apply minimal highpass to remove DC offset/drift
        if len(data) > 10:
            sos = signal.butter(2, 0.05, 'highpass', fs=target_sample_rate, output='sos')
            data = signal.sosfilt(sos, data)
        
        # Apply less time compression to preserve more signal character
        time_compression = 20  # Play 5x faster instead of 10x
        data = signal.resample(data, len(data) // time_compression)
        
        # Add back some noise
        noise_level = 0.02  # 2% noise
        noise = np.random.normal(0, noise_level, size=len(data))
        data = data + noise

        # Volume factor
        data = data*volume_factor
        
        # Re-normalize after adding noise
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        
        # Save as WAV file
        filename = f"{output_dir}/{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}.wav"
        wavfile.write(filename, target_sample_rate, data.astype(np.float32))
        print(f"Created sonification: {filename}")

# def create_sonification(stream, output_dir="sonification"):
#     """Create WAV files from seismic data with minimal filtering to preserve original characteristics"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for tr in stream:
#         # Get data
#         data = tr.data.copy()

#         # Normalize the data to [-1, 1] range
#         if np.max(np.abs(data)) > 0:
#             data = data / np.max(np.abs(data))

#         # Target sample rate for audio
#         target_sample_rate = 22050  # Common audio rate

#         # Resample to the target rate
#         samples_needed = int(len(data) * (target_sample_rate / tr.stats.sampling_rate))
#         data = signal.resample(data, samples_needed)

#         # Preserve more noise by avoiding heavy filtering
#         # Just apply minimal highpass to remove DC offset/drift
#         if len(data) > 10:
#             sos = signal.butter(2, 0.05, 'highpass', fs=target_sample_rate, output='sos')
#             data = signal.sosfilt(sos, data)

#         # Apply less time compression to preserve more signal character
#         time_compression = 5  # Play 5x faster instead of 10x
#         data = signal.resample(data, len(data) // time_compression)

#         # Save as WAV file
#         filename = f"{output_dir}/{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}.wav"
#         wavfile.write(filename, target_sample_rate, data.astype(np.float32))
#         print(f"Created sonification: {filename}")


def process_and_classify_all(base_dir, region):
    """Process all files, generate accelerograms and sonifications with classification"""
    # Process files for the specified region, keeping location codes separate
    region_dir = os.path.join(base_dir, region)
    location_streams = process_directory(region_dir)
    
    # Create output directories with region name
    accel_dir = f"accelerograms_{region}"
    sonic_dir = f"sonification_{region}"
    
    # Process each location code separately
    for location, streams_list in location_streams.items():
        print(f"\nProcessing location code: {location}")
        
        for stream in streams_list:
            # Calculate PGA/PGV and classify
            stream = calculate_pga_pgv(stream)
            stream = classify_by_frequency_content(stream)
            
            # Create accelerogram
            create_accelerogram(stream, accel_dir)
            
            # Create sonification
            create_sonification(stream, sonic_dir)
    
    print(f"\nAll processing complete for region: {region}")
    print(f"Accelerograms saved to: {accel_dir}")
    print(f"Sonifications saved to: {sonic_dir}")

# Main execution
if __name__ == "__main__":
    base_dir = '.'  # Adjust to your path
    regions = ['eastern-siberia', 'el-salvador', 'hawaii', 
               'honshu-japan', 'java-indonesia', 'north-honduras', 'puerto-rico',
               'myanmar', 'southern-alaska', 'taiwan-region']
    
    # Process a specific region
    # region = 'taiwan-region'  # Change to process different regions
    # process_and_classify_all(base_dir, region)
    
    # Uncomment to process all regions
    for region in regions:
        process_and_classify_all(base_dir, region)