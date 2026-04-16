import os
import subprocess
import sys
import shutil
import torch
import stat
from pathlib import Path

# ============================================================================
# SAFETY CONFIGURATION
# ============================================================================
# "Consumer Product" Philosophy:
# 1. NEVER upgrade existing packages (torch, numpy) implicitly.
# 2. Build against the currently active environment.
# 3. Install ONLY the final binary.
# ============================================================================

def on_rm_error(func, path, exc_info):
    """Handle read-only files on Windows during cleanup."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def run_command(cmd, cwd=None, env=None, check=True):
    """Run command with detailed logging."""
    print(f"[RUN] {cmd}")
    sys.stdout.flush()
    try:
        if check:
            subprocess.check_call(cmd, shell=True, cwd=cwd, env=env)
            return 0
        else:
            return subprocess.call(cmd, shell=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with error code {e.returncode}")
        if check:
            sys.exit(1)
        return e.returncode

def get_cuda_version():
    return torch.version.cuda

def check_compiler():
    nvcc_ok = False
    cl_ok = False
    try:
        subprocess.check_output("nvcc --version", shell=True, stderr=subprocess.STDOUT)
        nvcc_ok = True
    except: pass

    if shutil.which("cl.exe"):
        cl_ok = True
    else:
        try:
            subprocess.check_output("cl", shell=True, stderr=subprocess.STDOUT)
            cl_ok = True
        except: pass

    return nvcc_ok, cl_ok

def build_gsplat():
    print("\n==================================================")
    print("   Safe gsplat Installer (Surgical Mode)")
    print("==================================================")

    print(f"[OK] Python: {sys.version.split()[0]}")
    print(f"[OK] PyTorch: {torch.__version__}")

    cuda_ver = get_cuda_version()
    if not cuda_ver:
        print("[ERROR] CUDA not found! gsplat requires a CUDA-enabled PyTorch.")
        sys.exit(1)
    print(f"[OK] PyTorch CUDA: {cuda_ver}")

    nvcc_ok, cl_ok = check_compiler()

    script_dir = Path(__file__).parent.resolve()
    msvc_dir = script_dir / "portable_msvc"
    use_portable_msvc = False

    if not cl_ok:
        if (msvc_dir / "MSVC").exists() or (msvc_dir / "MSVC-Portable.bat").exists():
            use_portable_msvc = True

    print(f"[INFO] Compiler check:")
    print(f"   - NVCC: {'[OK] Found' if nvcc_ok else '[MISSING]'}")
    print(f"   - CL:   {'[OK] Found (System)' if cl_ok else ('[OK] Found (Portable)' if use_portable_msvc else '[MISSING]')}")

    if not nvcc_ok:
        print("\n[ERROR] NVCC (CUDA Compiler) is missing.")
        print("Please install the CUDA Toolkit that matches your PyTorch version.")
        sys.exit(1)

    if not cl_ok and not use_portable_msvc:
        print("\n[INFO] MSVC Compiler missing. Attempting to download Portable MSVC (600MB)...")
        try:
            run_command(f"git clone https://github.com/Delphier/MSVC {msvc_dir}")
            use_portable_msvc = True
        except:
            print("[ERROR] Failed to download compiler. Please install Visual Studio Build Tools manually.")
            sys.exit(1)

    # Check for pre-built wheel
    print("\n[INFO] Checking for pre-built wheel...")
    try:
        pt_tag = f"pt{torch.__version__.split('+')[0].replace('.', '')[:2]}"
        cu_tag = f"cu{cuda_ver.replace('.', '')}"
        index_url = f"https://docs.gsplat.studio/whl/{pt_tag}{cu_tag}"

        cmd = f"{sys.executable} -m pip install gsplat --index-url {index_url} --no-deps"
        if run_command(cmd, check=False) == 0:
            print("[OK] Installed from official wheel!")
            verify_install()
            return
    except:
        pass

    print("[INFO] Official wheel not found or incompatible. Proceeding to build from source...")

    # Clone source
    build_dir = script_dir / "gsplat_build"
    if not build_dir.exists():
        print(f"[INFO] Cloning gsplat source...")
        run_command(f"git clone --recursive https://github.com/nerfstudio-project/gsplat.git {build_dir}")
    else:
        print("[INFO] Source cache found.")

    # Build wheel
    print("\n[INFO] Compiling gsplat...")
    dist_dir = script_dir / "dist"
    dist_dir.mkdir(exist_ok=True)

    env = os.environ.copy()

    try: import ninja
    except:
        print("[INFO] Installing ninja build tool...")
        run_command(f"{sys.executable} -m pip install ninja")

    wheel_cmd = f"{sys.executable} -m pip wheel . -w {dist_dir} --verbose --no-build-isolation"

    if use_portable_msvc:
        msvc_installed = msvc_dir / "MSVC"
        vcvars = list(msvc_installed.rglob("vcvars64.bat"))

        activator = None
        if vcvars:
            activator = vcvars[0]
        else:
            print("[INFO] Initializing Portable MSVC...")
            subprocess.check_call(f'"{msvc_dir}/MSVC-Portable.bat"', shell=True, cwd=str(msvc_dir))
            vcvars = list(msvc_installed.rglob("vcvars64.bat"))
            if vcvars: activator = vcvars[0]

        if activator:
            full_cmd = f'"{activator}" && {wheel_cmd}'
        else:
            print("[ERROR] Compiler setup failed.")
            sys.exit(1)
    else:
        full_cmd = wheel_cmd

    run_command(full_cmd, cwd=str(build_dir), env=env)

    # Install wheel
    try:
        whl = sorted(list(dist_dir.glob("*.whl")), key=os.path.getmtime)[-1]
    except IndexError:
        print("[ERROR] Build failed to produce a .whl file.")
        sys.exit(1)

    print(f"\n[INFO] Installing {whl.name}...")
    install_cmd = f"{sys.executable} -m pip install {whl} --force-reinstall --no-deps"
    run_command(install_cmd)

    print("\n==================================================")
    print("[OK] SUCCESS")
    print("==================================================")
    verify_install()

def verify_install():
    try:
        import gsplat
        print(f"[OK] gsplat {gsplat.__version__} is importable.")
    except Exception as e:
        print(f"[WARN] Installed but import failed: {e}")

if __name__ == "__main__":
    build_gsplat()
