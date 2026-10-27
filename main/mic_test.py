import serial
import numpy as np
import time
from collections import deque

# config ------------------------------------------
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200            

SYNC_HEADER = b'\xaa\xbb\xcc\xdd'
SAMPLES_PER_BLOCK = 128
CHANNELS = 8
BYTES_PER_SAMPLE = 2  # int16
SAMPLE_RATE = 45000.0  # Hz
SPEED_OF_SOUND = 343.0

AUDIO_PACKET_SIZE = SAMPLES_PER_BLOCK * CHANNELS * BYTES_PER_SAMPLE # 2048 bytes

# change these before running 
MIC_POSITIONS = np.array([
    [0.097451, 0.022346, 0.0],        #1
    [0.07830, -0.062202, 0.0],        #2
    [0.0, -0.10, 0.0],              #3
    [-0.078066, -0.062496, 0.0],      #4
    [-0.097534, 0.022069, 0.0],       #5
    [0.0, 0.0, 0.0],                #6
    [-0.043558, 0.090179, 0.0],     #7
    [0.043219, 0.090179, 0.0],        #8
], dtype=np.float64)

SPHERE_POINTS = 500
SRP_WINDOW_SAMPLES = 1024
RMS_ACTIVITY_THRESHOLD = 40.0

#--------------------------------------------------

#Functions (copied from the https://github.com/SOUNDS-RESEARCH/DroneSynthesis2DOA), move into a separate file later if needed 

def calculate_rms(audio_matrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        rms = np.sqrt(np.mean(audio_matrix**2, axis=0))
        return np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)

def cart2sph(vector):
    # Calculate elevation angle, vector = [x,y,z]
    elevation_angle = np.degrees(np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2]))

    # Calculate azimuth angle
    azimuth_angle = np.degrees(np.arctan2(vector[1], vector[0]))
    azimuth_angle = (azimuth_angle + 360.0) if azimuth_angle < 0 else azimuth_angle
    return [elevation_angle, azimuth_angle]

# Create a grid of possible source directions
# Generate evenly distributed points on the spherical surface
# n is the amount of directions
def half_fibonacci_sphere(n):
    n *= 2

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    points = points[points[:, 2] >= 0]
    return np.transpose(points)


def build_srp(mic_positions, sample_rate, speed_of_sound, sphere_points_count):
    centered = mic_positions - np.mean(mic_positions, axis=0)
    sphere_points = half_fibonacci_sphere(sphere_points_count)

    mic_pairs = []
    max_delay_sec = 0.0
    for i in range(centered.shape[0]):
        for j in range(i + 1, centered.shape[0]):
            pair_vec = centered[i] - centered[j]
            mic_pairs.append((i, j, pair_vec))
            max_delay_sec = max(max_delay_sec, np.linalg.norm(pair_vec) / speed_of_sound)

    max_lag = int(np.ceil(max_delay_sec * sample_rate)) + 1
    delay_idx = np.empty((len(mic_pairs), sphere_points.shape[1]), dtype=np.int32)

    for pair_index, (_, _, pair_vec) in enumerate(mic_pairs):
        delay_sec = (sphere_points.T @ pair_vec) / speed_of_sound
        lag_samples = np.rint(delay_sec * sample_rate).astype(np.int32)
        delay_idx[pair_index] = np.clip(lag_samples + max_lag, 0, 2 * max_lag)

    return sphere_points, mic_pairs, delay_idx, max_lag


def estimate_doa_srp_phat(audio_matrix, mic_pairs, delay_idx, max_lag):
    frame = audio_matrix.astype(np.float64, copy=False)
    frame = frame - np.mean(frame, axis=0, keepdims=True)
    frame *= np.hanning(frame.shape[0])[:, None]

    n_samples = frame.shape[0]
    n_fft = 1 << (n_samples - 1).bit_length()
    spectrum = np.fft.rfft(frame, n=n_fft, axis=0)

    accumulator = np.zeros(delay_idx.shape[1], dtype=np.float64)
    eps = 1e-10
    for pair_index, (i, j, _) in enumerate(mic_pairs):
        cross_spectrum = spectrum[:, j] * np.conj(spectrum[:, i])
        phat = cross_spectrum / (np.abs(cross_spectrum) + eps)
        cc = np.fft.irfft(phat, n=n_fft)
        cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))
        accumulator += np.abs(cc[delay_idx[pair_index]])

    best_idx = int(np.argmax(accumulator))
    return best_idx, float(accumulator[best_idx])

#-------------------------------------------------

#main loop 

def main():
    print(f'Opening port {SERIAL_PORT}')
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f'Could not open port {SERIAL_PORT}: {e}')
        return

    # emtpy old buffers 
    ser.reset_input_buffer()
    last_print = time.time()
    rolling_buffer = deque(maxlen=SRP_WINDOW_SAMPLES)
    sphere_points, mic_pairs, delay_idx, max_lag = build_srp(
        MIC_POSITIONS,
        SAMPLE_RATE,
        SPEED_OF_SOUND,
        SPHERE_POINTS,
    )

    print('Starting audio stream.')
    
    try:
        while True:
            # check if header is present and read audio data
            if ser.read(4) == SYNC_HEADER:
                
                raw_audio = ser.read(AUDIO_PACKET_SIZE)
                if len(raw_audio) != AUDIO_PACKET_SIZE:
                    continue  # skip if we didnt get a full packet
                
                # raw data into array of int16
                audio_data = np.frombuffer(raw_audio, dtype=np.int16)
                if len(audio_data) != (CHANNELS * SAMPLES_PER_BLOCK):
                    continue

                audio_matrix = audio_data.reshape(CHANNELS, SAMPLES_PER_BLOCK).T
                rolling_buffer.extend(audio_matrix)
                    
                
                # Print every 200 ms (possibly increase)
                if time.time() - last_print > 0.2:
                    rms_levels = calculate_rms(audio_matrix)
                    mean_rms = float(np.mean(rms_levels))

                    doa_text = 'Az: ---- | El: ----'
                    if len(rolling_buffer) == SRP_WINDOW_SAMPLES and mean_rms >= RMS_ACTIVITY_THRESHOLD:
                        frame = np.array(rolling_buffer, dtype=np.float64)
                        doa_idx, confidence = estimate_doa_srp_phat(frame, mic_pairs, delay_idx, max_lag)
                        elevation_deg, azimuth_deg = cart2sph(sphere_points[:, doa_idx])
                        doa_text = (
                            f'Az: {azimuth_deg:6.1f} deg | '
                            f'El: {elevation_deg:6.1f} deg (S={confidence:8.1f})'
                        )
                    
                    if len(rms_levels) == 8:
                        print_str = ' | '.join([f'M{i+1}: {int(rms):5d}' for i, rms in enumerate(rms_levels)])
                        print(f'{print_str} | {doa_text}', end='\r')
                    last_print = time.time()

    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        ser.close()
        print('Port closed.')

if __name__ == '__main__':
    main()
