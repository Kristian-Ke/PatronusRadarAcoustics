import serial
import time


# ============================================================
# SETTINGS
# ============================================================

CLI_PORT = "/dev/ttyACM0"
BAUDRATE = 115200

CONFIG_FILE = "configs/profile_3d.cfg"


# ============================================================
# OPEN CLI PORT
# ============================================================

print("Opening CLI port...")

cli = serial.Serial(
    port=CLI_PORT,
    baudrate=BAUDRATE,
    timeout=1
)

time.sleep(1)

# Clear anything already sitting in the buffer
cli.reset_input_buffer()
cli.reset_output_buffer()

print("Connected to", CLI_PORT)
print()


# ============================================================
# SEND CONFIGURATION
# ============================================================

with open(CONFIG_FILE, "r") as f:

    for line in f:

        line = line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Ignore comments
        if line.startswith("%"):
            continue

        print(">>", line)

        # Send command
        cli.write(
            (line + "\n").encode()
        )

        cli.flush()

        # ----------------------------------------------------
        # Wait for radar response
        # ----------------------------------------------------

        time.sleep(0.1)

        response = b""

        # Read everything currently available
        while cli.in_waiting:

            response += cli.read(
                cli.in_waiting
            )

            time.sleep(0.01)

        response_text = response.decode(
            errors="ignore"
        )

        if response_text:
            print("<<", response_text.strip())

        # ----------------------------------------------------
        # Check for errors
        # ----------------------------------------------------

        if "Error" in response_text:

            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("RADAR RETURNED AN ERROR")
            print("Command:", line)
            print("Response:", response_text.strip())
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print()

            break


# ============================================================
# CLOSE
# ============================================================

cli.close()

print()
print("=" * 60)
print("Configuration finished")
print("=" * 60)