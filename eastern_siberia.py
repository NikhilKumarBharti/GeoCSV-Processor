import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob2 as glob
import os
from obspy import UTCDateTime
from obspy.core import Stream, Trace, Stats
import sounddevice as sd
import scipy.signal as signal

def read_geocsv(file_path):
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
    
    # Properly handle SID with variable number of parts
    parts = sid.split('_')
    if len(parts) >= 3:
        network = parts[0]
        station = parts[-1]
        channel = parts[-2]  # Last part is usually the channel
    else:
        network = ''
        station = sid
        channel = ''
    
    sample_rate = float(metadata.get('sample_rate_hz', 1.0))
    start_time = UTCDateTime(metadata.get('start_time', '2000-01-01T00:00:00Z'))
    
    # Create ObsPy Stats object
    stats = Stats({
        'network': network,
        'station': station,
        'channel': channel,
        'starttime': start_time,
        'sampling_rate': sample_rate,
        'npts': len(data)
    })
    
    # Create ObsPy Trace with the sample data
    tr = Trace(data=np.array(data[data_column]), header=stats)
    
    return Stream(traces=[tr])

def counts_to_acceleration(stream, sensitivity=1.0):
    """
    Convert counts to acceleration.
    
    Parameters:
    - stream: ObsPy Stream object
    - sensitivity: Instrument sensitivity in counts/(m/s^2)
    
    Returns:
    - Stream object with data in m/s^2
    """
    for tr in stream:
        # Apply instrument correction (simplified)
        tr.data = tr.data / sensitivity
        tr.stats.units = 'm/s^2'
    
    return stream

def process_directory(directory):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    streams = []
    for file in csv_files:
        print(f"Processing {file}")
        stream = read_geocsv(file)
        # Apply conversion to acceleration
        accel_stream = counts_to_acceleration(stream, sensitivity=2800.0)  # Example value
        streams.append(accel_stream)
    
    return streams

def get_data_range(streams):
    """Find the global min and max values across all streams"""
    global_min = float('inf')
    global_max = float('-inf')
    
    for stream in streams:
        for tr in stream:
            min_val = np.min(tr.data)
            max_val = np.max(tr.data)
            
            if min_val < global_min:
                global_min = min_val
            if max_val > global_max:
                global_max = max_val
    
    # Add a small buffer (10%) to the range
    range_size = global_max - global_min
    buffer = range_size * 0.1
    
    return global_min - buffer, global_max + buffer

def plot_all_streams_with_standard_range(streams, save_dir='accelerograms'):

    os.makedirs(save_dir, exist_ok=True)
    """Plot all streams with a standardized y-axis range"""
    # Get the global data range
    y_min, y_max = get_data_range(streams)
    
    # Plot each stream
    for i, stream in enumerate(streams):
        fig = plt.figure(figsize=(12, 6))
        for tr in stream:
            times = tr.times() + (tr.stats.starttime.timestamp - tr.stats.starttime.timestamp)
            plt.plot(times, tr.data, label=f"{tr.stats.channel}")
        
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.title(f'Accelerogram - {os.path.basename(stream[0].stats.station)} ({stream[0].stats.channel})')
        plt.ylim(y_min, y_max)  # Set the standardized y-axis range
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Define the filename and save path
        filename = f"accelerogram_{stream[0].stats.station}_{stream[0].stats.channel}.png"
        save_path = os.path.join(save_dir, filename)

        # Save the figure
        plt.savefig(save_path, dpi=300)
        plt.close()  # Close the figure to free memory
        
    print(f"All accelerograms plotted with y-axis range: [{y_min:.4f}, {y_max:.4f}]")

def play_seismic_audio(stream, scale_factor=0.5, duration=10):
    """
    Convert seismic data to audio and play it through speakers.
    
    Parameters:
    - stream: ObsPy Stream object
    - scale_factor: Factor to scale the amplitude (0-1)
    - duration: Duration in seconds to play (will resample if needed)
    """
    for tr in stream:
        # Get sample rate and data
        original_sample_rate = int(tr.stats.sampling_rate)
        data = tr.data.copy()
        
        # Normalize the data to [-1, 1] range for audio
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data)) * scale_factor
        
        # Use a standard audio sample rate
        target_sample_rate = 44100  # Standard audio rate
        
        # Resample to standard audio rate
        samples_needed = int(len(data) * (target_sample_rate / original_sample_rate))
        data = signal.resample(data, samples_needed)
        
        # Apply a bandpass filter to focus on audible frequencies
        nyquist = 0.5 * target_sample_rate
        low = 20 / nyquist
        high = min(10000, nyquist * 0.95) / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        data = signal.filtfilt(b, a, data)
        
        # Limit duration if needed
        if len(data) > duration * target_sample_rate:
            data = data[:int(duration * target_sample_rate)]
        
        # Play the audio
        print(f"Playing audio for {tr.stats.station}.{tr.stats.channel}...")
        try:
            sd.play(data, samplerate=target_sample_rate)
            sd.wait()  # Wait until the audio is done playing
        except Exception as e:
            print(f"Error playing audio: {e}")
            print("Attempting to use a different sample rate...")
            try:
                # Try a different common sample rate
                sd.play(data, samplerate=22050)
                sd.wait()
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                print("Saving audio to file instead...")
                try:
                    from scipy.io import wavfile
                    wavfile.write(f"{tr.stats.station}_{tr.stats.channel}.wav", 
                                 22050, data.astype(np.float32))
                    print(f"Audio saved to {tr.stats.station}_{tr.stats.channel}.wav")
                except Exception as e3:
                    print(f"Could not save audio: {e3}")
        print("Audio playback complete.")

def play_all_streams(streams, scale_factor=0.5, duration=10):
    """Play audio for all streams"""
    for i, stream in enumerate(streams):
        print(f"\nPlaying stream {i+1} of {len(streams)}:")
        play_seismic_audio(stream, scale_factor, duration)
        
        # Ask user before playing the next stream
        if i < len(streams) - 1:
            input("Press Enter to play the next stream...")

def create_seismic_sonification(streams, output_dir="sonification"):
    """
    Create WAV files from seismic data without attempting to play them directly.
    """
    from scipy.io import wavfile
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, stream in enumerate(streams):
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
            
            # Apply frequency shift to make the signal more audible
            # (Optional) Time compression - speeds up playback
            time_compression = 10  # Play 10x faster
            data = signal.resample(data, len(data) // time_compression)
            
            # Save as WAV file
            filename = f"{output_dir}/{tr.stats.station}_{tr.stats.channel}.wav"
            wavfile.write(filename, target_sample_rate, data.astype(np.float32))
            print(f"Created sonification: {filename}")
    
    print(f"All sonification files created in directory: {output_dir}")

# Main processing
base_dir = '.'  # Adjust to your path
region = 'java-indonesia'  # Or choose another region

# Process files for a specific region
region_dir = os.path.join(base_dir, region)
streams = process_directory(region_dir)

# Plot all streams with standardized range
plot_all_streams_with_standard_range(streams, region_dir)

# Play audio for all streams
# play_all_streams(streams, scale_factor=0.5, duration=10)

# Sonification
create_seismic_sonification(streams, region_dir)