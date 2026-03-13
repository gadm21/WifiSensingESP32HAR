#!/usr/bin/env python3
"""
CSI collector for two-ESP32 setup (sender + receiver).
- Only logs CSI from the *receiver* serial port.
- Saves raw CSI lines with the required header (no extra timestamps).
- Prints live sampling rate (CSI lines/sec).
- Runs for a fixed duration, then exits.

Run using: python collect.py --rx-port /dev/ttyACM1 --tx-port /dev/ttyACM0 --out data/csi_raw13.csv --duration 1000 --times 3
python collect.py   --rx-port /dev/cu.usbmodem1301   --tx-port /dev/cu.usbmodem1201   --out mac_data/train/work.csv   --duration 1000 --times 3
"""

import argparse
import sys
import time
import threading
import signal

try:
    import serial
    import serial.tools.list_ports as list_ports
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial", file=sys.stderr)
    sys.exit(1)

HEADER = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"

stop_event = threading.Event()
_csi_count = 0               # total CSI lines written
_csi_count_lock = threading.Lock()

def reset_board(port, baud, label):
    """Open/close briefly to toggle DTR/RTS which resets many ESP32 boards."""
    try:
        with serial.Serial(port, baudrate=baud) as s:
            s.dtr = False; s.rts = True
            time.sleep(0.05)
            s.dtr = True;  s.rts = False
            time.sleep(0.05)
        print(f"[info] Reset pulsed on {label} at {port}")
    except Exception as e:
        print(f"[warn] Could not reset {label} at {port}: {e}", file=sys.stderr)

def reader_thread(rx_port, baud, outfile, duration):
    """Read receiver serial and write raw CSI lines to CSV for `duration` seconds."""
    global _csi_count
    end_time = time.time() + duration
    try:
        with serial.Serial(rx_port, baudrate=baud, timeout=0.05) as ser, open(outfile, "w", buffering=1) as f:
            # Write the required header once
            f.write(HEADER + "\n")

            # Drain any boot garbage quickly
            _ = ser.read(ser.in_waiting or 1)

            while not stop_event.is_set() and time.time() < end_time:
                line = ser.readline()
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                # Only capture raw CSI rows (they already start with "CSI_DATA,")
                if text.startswith("CSI_DATA,"):
                    f.write(text + "\n")
                    with _csi_count_lock:
                        _csi_count += 1

    except serial.SerialException as e:
        print(f"[error] Serial error on {rx_port}: {e}", file=sys.stderr)
    except OSError as e:
        print(f"[error] OS error while accessing {rx_port}: {e}", file=sys.stderr)

def rate_monitor_thread():
    """Print live CSI sampling rate (lines/sec) every second."""
    last_count = 0
    run_start = time.time()
    last_time = run_start
    
    while not stop_event.is_set():
        time.sleep(1.0)
        current_time = time.time()
        with _csi_count_lock:
            current_count = _csi_count
        
        # Calculate rate based on actual time elapsed since last print
        time_elapsed = current_time - last_time
        if time_elapsed > 0:
            cps = int((current_count - last_count) / time_elapsed)
            total_elapsed = current_time - run_start
            print(f"[rate] {cps:6d} CSI/s   (total={current_count}, elapsed={total_elapsed:5.1f}s)")
        
        last_count = current_count
        last_time = current_time

def discover_note(port_hint):
    """Utility text to help user see connected ports."""
    ports = [p.device for p in list_ports.comports()]
    hint = f"Available ports: {', '.join(ports) or 'none found'}"
    if port_hint and port_hint not in ports:
        hint += f"\n[warn] '{port_hint}' not detected right now."
    return hint

def main():
    ap = argparse.ArgumentParser(description="Collect raw CSI from receiver ESP32 for a fixed duration.")
    ap.add_argument("--rx-port", required=True, help="Serial port of the receiver ESP32 (e.g. /dev/ttyACM1 or COM5)")
    ap.add_argument("--tx-port", default=None, help="(Optional) Serial port of the sender ESP32 (only reset pulse, no logging)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--out", required=True, help="Output CSV file path (without extension)")
    ap.add_argument("--duration", type=float, required=True, help="Recording duration in seconds")
    ap.add_argument("--times", type=int, default=1, help="Number of times to repeat the collection (default: 1)")
    ap.add_argument("--no-reset", action="store_true", help="Skip reset pulses on the boards")
    args = ap.parse_args()

    print(f"[info] Receiver: {args.rx_port} @ {args.baud}")
    if args.tx_port:
        print(f"[info] Sender  : {args.tx_port} @ {args.baud} (no logging, optional reset only)")
    print(f"[info] Base output: {args.out}")
    print(f"[info] Duration: {args.duration:.3f} s")
    print(f"[info] Repeats: {args.times}")
    print(discover_note(args.rx_port))

    # Optional reset pulses to (re)start apps on both boards
    if not args.no_reset:
        reset_board(args.rx_port, args.baud, "receiver")
        if args.tx_port:
            reset_board(args.tx_port, args.baud, "sender")

    for i in range(args.times):
        global _csi_count
        _csi_count = 0  # Reset counter for each run
        stop_event.clear()  # Reset the stop event
        
        # Add sequence number to output filename
        base_name = args.out.replace('.csv', '')
        output_file = f"{base_name}_{i+1}.csv"
        
        print(f"\n[info] Starting collection {i+1}/{args.times} -> {output_file}")
        
        # Threads: serial reader + live rate monitor
        t_reader = threading.Thread(
            target=reader_thread, 
            args=(args.rx_port, args.baud, output_file, args.duration), 
            daemon=True
        )
        t_rate = threading.Thread(target=rate_monitor_thread, daemon=True)

        def handle_sigint(signum, frame):
            stop_event.set()
        signal.signal(signal.SIGINT, handle_sigint)

        t_reader.start()
        t_rate.start()

        # Wait until duration elapsed or Ctrl-C
        deadline = time.time() + args.duration
        try:
            while t_reader.is_alive():
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(0.2, max(0.0, remaining)))
        finally:
            stop_event.set()
            t_reader.join()
            # rate thread is daemon; give it a moment to print final line if desired
            time.sleep(0.05)

    print("\n[info] All collections completed.")

if __name__ == "__main__":
    main()
