# Raspberry Pi Setup Guide

This guide covers setting up and running the Smart Aid stream server on a Raspberry Pi.

## Prerequisites

- Raspberry Pi (tested on Pi 4/5)
- Pi Camera Module (IMX219 or compatible)
- Raspberry Pi OS (Bookworm or later)
- Network connection (same network as MacBook)

## SSH Configuration

### On MacBook

1. Add Pi to SSH config (`~/.ssh/config`):

```
Host bonny
    HostName bonny.local
    User bonny
```

2. Copy SSH key to Pi (one-time setup):

```bash
ssh-copy-id bonny
```

Enter password when prompted.

3. Test connection:

```bash
ssh bonny "hostname && uname -a"
```

## Pi Setup Commands

### 1. Enable Camera

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot when prompted
```

### 2. Install Dependencies

```bash
sudo apt update
sudo apt install -y python3-opencv python3-flask python3-picamera2
```

### 3. Copy Project Files (from MacBook)

```bash
# From MacBook terminal
scp -r /path/to/smart-aid/pi_scripts bonny:~/smart-aid/
```

Or clone the repository on the Pi:

```bash
git clone <repo-url> ~/smart-aid
```

## Running the Stream Server

### Start Server (Foreground)

```bash
cd ~/smart-aid
python3 pi_scripts/stream_server.py
```

### Start Server (Background)

```bash
cd ~/smart-aid
nohup python3 pi_scripts/stream_server.py > /tmp/stream.log 2>&1 &
```

### Check Server Status

```bash
# View logs
cat /tmp/stream.log

# Check if running
ps aux | grep stream_server

# Get Pi IP address
hostname -I
```

### Stop Server

```bash
pkill -f stream_server.py
```

## Server Options

```bash
python3 pi_scripts/stream_server.py --help

# Options:
#   --port PORT       Server port (default: 5000)
#   --width WIDTH     Frame width (default: 640)
#   --height HEIGHT   Frame height (default: 480)
#   --fps FPS         Target FPS (default: 30)
#   --camera TYPE     Camera backend: opencv or picamera2 (default: picamera2)
```

### Examples

```bash
# Higher resolution
python3 pi_scripts/stream_server.py --width 1280 --height 720

# Different port
python3 pi_scripts/stream_server.py --port 8080

# Use OpenCV backend
python3 pi_scripts/stream_server.py --camera opencv
```

## Accessing the Stream

Once the server is running, you'll see output like:

```
Starting server on port 5000
Resolution: 640x480
Stream URL: http://0.0.0.0:5000/video_feed
 * Running on http://192.168.1.166:5000
```

### From Browser

Open: `http://<PI_IP>:5000/`

### From MacBook (Smart Aid)

```bash
cd smart-aid
source venv/bin/activate
python src/main.py --source http://<PI_IP>:5000/video_feed
```

### Health Check

```bash
curl http://<PI_IP>:5000/health
# Returns: {"status": "ok", "camera": true}
```

## Auto-Start on Boot (Optional)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/smart-aid-stream.service
```

Add:

```ini
[Unit]
Description=Smart Aid Camera Stream
After=network.target

[Service]
Type=simple
User=bonny
WorkingDirectory=/home/bonny/smart-aid
ExecStart=/usr/bin/python3 /home/bonny/smart-aid/pi_scripts/stream_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable smart-aid-stream
sudo systemctl start smart-aid-stream

# Check status
sudo systemctl status smart-aid-stream
```

## Troubleshooting

### Camera Not Found

```bash
# Check if camera is detected
libcamera-hello --list-cameras

# If not detected, check connections and run:
sudo raspi-config
# Interface Options → Camera → Enable
```

### Module Not Found Errors

```bash
# Reinstall dependencies
sudo apt install --reinstall python3-opencv python3-flask python3-picamera2
```

### Permission Denied

```bash
# Add user to video group
sudo usermod -aG video $USER
# Log out and back in
```

### Connection Refused

```bash
# Check if server is running
ps aux | grep stream_server

# Check firewall
sudo ufw status
sudo ufw allow 5000/tcp  # If enabled
```

## Current Configuration

| Setting | Value |
|---------|-------|
| Pi Hostname | `bonny.local` |
| Pi User | `bonny` |
| Pi IP | `192.168.1.166` |
| Stream URL | `http://192.168.1.166:5000/video_feed` |
| Camera | IMX219 (Pi Camera Module v2) |
| Resolution | 640x480 |
