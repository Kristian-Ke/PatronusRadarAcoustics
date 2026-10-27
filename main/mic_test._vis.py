import serial
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

# config ------------------------------------------
SERIAL_PORT = '/dev/ttyACM0'  
BAUD_RATE = 115200            

SYNC_HEADER = b'\xaa\xbb\xcc\xdd'
SAMPLES_PER_BLOCK = 128
CHANNELS = 8
BYTES_PER_SAMPLE = 2  # int16
SAMPLE_RATE = 44117.65
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

# functions, move into a separate file later

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


def init_visualizer(mic_positions, sample_rate, fft_size, spec_frames=220):
    centered = mic_positions - np.mean(mic_positions, axis=0)
    span = np.maximum(np.ptp(centered, axis=0), 1e-3)
    max_span = float(np.max(span))
    arrow_scale = 0.7 * max_span

    plt.ion()
    fig = plt.figure('Mic Array + DOA Visualizer', figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 1.0])

    ax_array = fig.add_subplot(gs[:, 0], projection='3d')
    ax_doa = fig.add_subplot(gs[:, 1], projection='polar')

    ax_array.scatter(centered[:, 0], centered[:, 1], centered[:, 2], c='tab:blue', s=45)
    for i, (x, y, z) in enumerate(centered):
        ax_array.text(x, y, z, f'M{i+1}', fontsize=8)

    lim = 0.9 * max_span
    ax_array.set_xlim(-lim, lim)
    ax_array.set_ylim(-lim, lim)
    ax_array.set_zlim(-lim, lim)
    ax_array.set_xlabel('X (m)')
    ax_array.set_ylabel('Y (m)')
    ax_array.set_zlabel('Z (m)')
    ax_array.set_title('Microphone Array (centered)')
    doa_quiver = ax_array.quiver(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, color='tab:red', linewidth=2)


    ax_doa.set_theta_zero_location('E')
    ax_doa.set_theta_direction(-1)
    ax_doa.set_rlim(0, 90)
    ax_doa.set_title('DOA (Azimuth + Elevation)')
    doa_line, = ax_doa.plot([0.0, 0.0], [0.0, 0.0], color='tab:red', linewidth=2)
    doa_dot, = ax_doa.plot([0.0], [0.0], marker='o', color='tab:red', markersize=7)
    doa_text = ax_doa.text(0.05, 0.02, 'Az: ---- deg | El: ---- deg', transform=ax_doa.transAxes)

    fig.tight_layout()
    return {
        'fig': fig,
        'ax_array': ax_array,
        'doa_quiver': doa_quiver,
        'arrow_scale': arrow_scale,
        'fft_size': fft_size,
        'fft_window': np.hanning(fft_size),
        'doa_line': doa_line,
        'doa_dot': doa_dot,
        'doa_text': doa_text,
    }


def update_visualizer(viz, audio_matrix, doa_vector=None, azimuth_deg=None, elevation_deg=None):
    mono = np.mean(audio_matrix, axis=1).astype(np.float64, copy=False)
    spectrum = np.fft.rfft(mono * viz['fft_window'], n=viz['fft_size'])
    db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-8))

    if doa_vector is not None:
        if viz['doa_quiver'] is not None:
            viz['doa_quiver'].remove()
        direction = doa_vector * viz['arrow_scale']
        viz['doa_quiver'] = viz['ax_array'].quiver(
            0.0,
            0.0,
            0.0,
            direction[0],
            direction[1],
            direction[2],
            color='tab:red',
            linewidth=2,
        )

    if azimuth_deg is not None and elevation_deg is not None:
        theta = np.deg2rad(azimuth_deg)
        radius = float(np.clip(elevation_deg, 0.0, 90.0))
        viz['doa_line'].set_data([theta, theta], [0.0, radius])
        viz['doa_dot'].set_data([theta], [radius])
        viz['doa_text'].set_text(f'Az: {azimuth_deg:6.1f} deg | El: {elevation_deg:6.1f} deg')
    else:
        viz['doa_line'].set_data([0.0, 0.0], [0.0, 0.0])
        viz['doa_dot'].set_data([0.0], [0.0])
        viz['doa_text'].set_text('Az: ---- deg | El: ---- deg')

    viz['fig'].canvas.draw_idle()
    plt.pause(0.001)

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

    viz = None
    viz = init_visualizer(MIC_POSITIONS, SAMPLE_RATE, SAMPLES_PER_BLOCK)
    print('Starting visualizer')
    
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
                    doa_vector = None
                    azimuth_deg = None
                    elevation_deg = None
                    if len(rolling_buffer) == SRP_WINDOW_SAMPLES and mean_rms >= RMS_ACTIVITY_THRESHOLD:
                        frame = np.array(rolling_buffer, dtype=np.float64)
                        doa_idx, confidence = estimate_doa_srp_phat(frame, mic_pairs, delay_idx, max_lag)
                        elevation_deg, azimuth_deg = cart2sph(sphere_points[:, doa_idx])
                        doa_vector = sphere_points[:, doa_idx]
                        doa_text = (
                            f'Az: {azimuth_deg:6.1f} deg | '
                            f'El: {elevation_deg:6.1f} deg (S={confidence:8.1f})'
                        )
                    
                    if len(rms_levels) == 8:
                        print_str = ' | '.join([f'M{i+1}: {int(rms):5d}' for i, rms in enumerate(rms_levels)])
                        print(f'{print_str} | {doa_text}', end='\r')

                    if viz is not None:
                        if not plt.fignum_exists(viz['fig'].number):
                            print('\nVisualizer window closed. Stopping.')
                            break
                        update_visualizer(viz, audio_matrix, doa_vector, azimuth_deg, elevation_deg)

                    last_print = time.time()

    except KeyboardInterrupt:
        print('\nStopping.')
    finally:
        ser.close()
        plt.close('all')
        print('Port closed.')

if __name__ == '__main__':
    main()
