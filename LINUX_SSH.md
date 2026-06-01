# FlightPath — Linux SSH Guide

This guide covers running FlightPath on a Linux GPU server over SSH, with the annotated video saved as a file that you can watch on your Windows machine.

---

## Prerequisites on the Linux Server

| Requirement | Minimum version |
|---|---|
| GCC / g++ | 9+ (for C++17) |
| CMake | 3.15+ |
| OpenCV | 4.x (with CUDA support recommended) |
| CUDA Toolkit | 11+ |
| NVIDIA driver | 450+ |

---

## Step 1 — Clone the Repo on the Server

```bash
git clone https://github.com/axli16/FlightPath.git
cd FlightPath
```

Or, if you already have it locally, just `git push` and `git pull` on the server.

---

## Step 2 — Check Dependencies

Run the setup checker. It will verify all tools are present and download YOLO model files if needed:

```bash
bash scripts/setup_linux.sh
```

> [!NOTE]
> If your OpenCV was built from source with CUDA support, skip the package manager install step.
> The script only auto-installs the system package (no CUDA). For CUDA-accelerated OpenCV,
> build it from source: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

---

## Step 3 — Build

```bash
bash scripts/build.sh
```

The binary is placed at: `build/bin/FlightPath`

---

## Step 4 — Put Your Video on the Server

**Option A — SCP from Windows (PowerShell):**
```powershell
scp data\dashcam.mp4 user@linuxserver:~/FlightPath/data/dashcam.mp4
```

**Option B — Copy from a network mount or USB drive already on the server.**

---

## Step 5 — Run Headless (No Display)

The `--no-display` flag skips the preview window. Always pair it with `--output` so the result is saved:

```bash
./build/bin/FlightPath data/dashcam.mp4 \
  --cuda \
  --no-display \
  --output result.mp4
```

> [!TIP]
> Run inside `tmux` or `screen` so the job continues if your SSH session drops:
> ```bash
> tmux new -s flightpath
> ./build/bin/FlightPath data/dashcam.mp4 --cuda --no-display --output result.mp4
> # Detach: Ctrl+B then D
> # Re-attach later: tmux attach -t flightpath
> ```

Other useful flags:
```bash
--frameskip 2        # Process every 2nd frame (faster)
--input-size 416     # Larger detection input (more accurate, slower)
--confidence 0.4     # Lower threshold to catch more objects
--no-road            # Skip UFLDv2 road model if not available
```

---

## Step 6 — Copy the Result Back to Windows

In PowerShell on your Windows machine:

```powershell
scp user@linuxserver:~/FlightPath/result.mp4 .\result.mp4
```

Then open `result.mp4` in VLC, Windows Media Player, or any video player.

---

## Optional — Live Display via X11 Forwarding

If you want the real-time preview window on your Windows machine, you need an X server:

1. **Install VcXsrv** on Windows: https://sourceforge.net/projects/vcxsrv/
2. Launch XLaunch → *Multiple windows* → *Start no client* → **check "Disable access control"**
3. SSH with X forwarding:
   ```bash
   ssh -X user@linuxserver
   ```
4. Run without `--no-display` (the program auto-detects `$DISPLAY`):
   ```bash
   ./build/bin/FlightPath data/dashcam.mp4 --cuda
   ```

> [!NOTE]
> X11 forwarding works but adds latency. For any serious processing, the **save-to-file** workflow (Step 5) is much faster.

---

## Troubleshooting

### `std::thread` linker errors
Make sure you have `Threads::Threads` in CMakeLists.txt (already done). If you see:
```
undefined reference to `pthread_create`
```
Add `-lpthread` to your compile command, or re-run `bash scripts/build.sh`.

### OpenCV `imshow` crash with "cannot connect to X server"
You forgot `--no-display`. Either:
- Add `--no-display` to your command, **or**
- The auto-detect will kick in automatically when `$DISPLAY` is not set

### CUDA not available at runtime
```bash
# Verify GPU is visible
nvidia-smi

# Check that OpenCV was built with CUDA
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Should print 1 or more
```

### CUDA not found in PATH during setup check
If the setup script shows a `[!] CUDA not found in PATH` warning, the shell doesn't know where `nvcc` or the CUDA libraries are located. Run the following commands to add them to your environment permanently:
```bash
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
source ~/.bashrc
```

### `Failed to initialize NVML: Driver/library version mismatch`
This happens when your package manager installs an updated NVIDIA driver in the background, making the active kernel module version differ from the user-space libraries. 
- **Solution A (Recommended)**: Simply reboot the server to load the new kernel module:
  ```bash
  sudo reboot
  ```
- **Solution B (Without Rebooting)**: If you cannot reboot, unload and reload the kernel modules:
  ```bash
  # Stop any processes using the GPU (docker, python, etc.)
  sudo rmmod nvidia_uvm
  sudo rmmod nvidia_modeset
  sudo rmmod nvidia
  # Verify mismatch is resolved
  nvidia-smi
  ```

### `cannot execute binary file`
This error occurs if you try to execute the compiled C++ executable with the `bash` command prefix (e.g., `bash ./build/bin/FlightPath`). `FlightPath` is a compiled machine-code binary (ELF executable), not a plain-text shell script. 
- **Solution**: Run the executable directly without the `bash` prefix:
  ```bash
  ./build/bin/FlightPath data/dashcam.mp4 --cuda --no-display --output result.mp4
  ```

### `OpenCV: The function is not implemented (cvDestroyAllWindows)`
This happens on Linux servers where OpenCV was built from source without highgui windowing GUI support (like GTK+ or Cocoa). Calling `cv::destroyAllWindows()` directly in that environment crashes the application.
- **Solution**: I have wrapped the call in a `try-catch` block inside `src/VideoProcessor.cpp`. Rebuild the program on your server by running:
  ```bash
  bash scripts/build.sh
  ```

### GoPro H.265 (HEVC) or High Resolution Video Mismatches
If your video file is a high-resolution GoPro recording (e.g. 4000x3000 at 100 Mbps using **H.265 / HEVC** codec), many standard Linux OpenCV builds will fail to open it due to missing proprietary decoding codecs or because the resolution is too large to load in standard CPU buffers.
- **Solution**: Convert and downscale the video to standard H.264 (AVC) at 1080p using `ffmpeg`. This guarantees universal support and speeds up your processing frame rates significantly:
  ```bash
  ffmpeg -i data/dashcam.mp4 -c:v libx264 -preset fast -crf 23 -vf "scale=1920:-2" -c:a copy data/dashcam_converted.mp4
  ```
  Then run the path planner using the converted video:
  ```bash
  ./build/bin/FlightPath data/dashcam_converted.mp4 --cuda --no-display --output result.mp4
  ```

### Low performance / slow inference
- Make sure `--cuda` is passed
- Check `nvidia-smi` shows GPU utilization > 0% during processing
- Try `--frameskip 3` to process fewer frames
- Reduce `--input-size 224` for faster but less accurate detection

---

## Step 7 — Remote Debugging with VS Code (Option A)

This section explains how to use VS Code on Windows to debug the C++ application running on the remote Linux GPU server with interactive breakpoints.

### 1. Prerequisite: Passwordless SSH Setup (Optional but highly recommended)
To avoid typing your password every time VS Code connects or runs tasks:
- **On Windows (PowerShell):**
  1. Generate SSH keys if you haven't already:
     ```powershell
     ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519"
     ```
     *(Press Enter to bypass passphrase prompt)*
  2. Copy the public key to the remote server:
     ```powershell
     cat ~\.ssh\id_ed25519.pub | ssh alixer@100.93.71.27 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
     ```

### 2. VS Code Extensions Setup
1. In VS Code on Windows, install the **Remote - SSH** extension (from Microsoft).
2. Click the green button in the bottom-left corner of VS Code (or press `Ctrl+Shift+P` and type `Remote-SSH: Connect to Host...`).
3. Select **Add New SSH Host...** and enter:
   ```bash
   ssh alixer@100.93.71.27
   ```
4. Connect to the host. When prompted, select **Linux** as the operating system and enter your password.
5. Once the remote workspace is loaded:
   - Go to the Extension marketplace (`Ctrl+Shift+X`).
   - You will see a section: **SSH: 100.93.71.27 - INSTALLED**.
   - Install the **C/C++** and **CMake Tools** extensions *on the SSH host*.

### 3. Server Dependencies
Ensure GDB (GNU Debugger) is installed on the remote Linux machine:
```bash
sudo apt update && sudo apt install -y gdb
```

### 4. Build with Debug Symbols
To hit breakpoints and inspect variables, you must build the binary in **Debug** mode:
```bash
bash scripts/build.sh --debug
```

### 5. Launch the Debugger
1. In your remote VS Code window, open the `FlightPath` workspace folder (`/home/alixer/FlightPath`).
2. Open any C++ source file, such as `src/VideoProcessor.cpp`, and set a breakpoint by clicking to the left of the line numbers.
3. Switch to the **Run & Debug** view (`Ctrl+Shift+D`).
4. Select **(gdb) Launch FlightPath (Remote SSH)** from the dropdown.
5. Press **F5** (or click the green play button).
   - This automatically runs the pre-launch task to verify the Debug build.
   - It runs the program headlessly using the configured input/output parameters.
   - You can step through execution, hover to inspect values, and use the debug console!

---

## Quick Reference

```bash
# Full headless run with CUDA, save to file
./build/bin/FlightPath data/dashcam.mp4 --cuda --no-display --output result.mp4

# Copy result to Windows (run this in PowerShell on Windows)
scp user@192.168.1.x:~/FlightPath/result.mp4 .\result.mp4
```
