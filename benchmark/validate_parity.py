import subprocess
import re

def extract_pixels(output, prefix, is_cl=False):
    pixels = []
    if is_cl:
        # OpenCL format: Prefix - 20% pixel: [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        pattern = prefix + r" - \d+% pixel: \[([\d.eE+-]+), ([\d.eE+-]+), ([\d.eE+-]+), ([\d.eE+-]+)\]"
    else:
        # Mojo format: Prefix - 20 % pixel: [ 0.0 0.0 0.0 1.0 ]
        pattern = prefix + r" - \d+ % pixel: \[ ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+) ([\d.eE+-]+) \]"
    
    for match in re.finditer(pattern, output):
        pixels.append([float(x) for x in match.groups()])
    return pixels if pixels else None

def get_mojo_gpu_output():
    print("Running Mojo GPU benchmark...")
    result = subprocess.run(["pixi", "run", "mojo", "sigmoid_benchmark_gpu.mojo"], 
                           capture_output=True, text=True, cwd=".")
    if result.returncode != 0:
        print("Mojo GPU Error:", result.stderr)
        return None, None
    rgb = extract_pixels(result.stdout, "RGB Ratio")
    per = extract_pixels(result.stdout, "Per Channel")
    return rgb, per

def get_opencl_output():
    print("Running OpenCL parity check...")
    # Using clang as per the new project standard
    subprocess.run(["clang", "parity_check.c", "-o", "parity_check", "-lOpenCL", "-lm"], cwd=".", capture_output=True)
    result = subprocess.run(["./parity_check"], capture_output=True, text=True, cwd=".")
    if result.returncode != 0:
        print("OpenCL Error:", result.stderr)
        return None, None
    rgb = extract_pixels(result.stdout, "RGB Ratio", is_cl=True)
    per = extract_pixels(result.stdout, "Per Channel", is_cl=True)
    return rgb, per

def compare_pixels(name, list1, list2, tolerance=1e-6):
    if list1 is None or list2 is None:
        print(f"❌ FAIL {name}: Missing data")
        return False
    if len(list1) != len(list2):
        print(f"❌ FAIL {name}: Data length mismatch")
        return False
    
    all_match = True
    for i, (val1, val2) in enumerate(zip(list1, list2)):
        pixel_match = True
        for v1, v2 in zip(val1, val2):
            if abs(v1 - v2) > tolerance:
                pixel_match = False
                all_match = False
                break
        if not pixel_match:
            print(f"  Mismatch at { (i+1)*20 }% pixel:")
            print(f"    Got (Mojo GPU): {val1}")
            print(f"    Expected (OpenCL): {val2}")
            
    status = "✅ PASS" if all_match else "❌ FAIL"
    print(f"{status} {name}")
    return all_match

def main():
    mojo_gpu_rgb, mojo_gpu_per = get_mojo_gpu_output()
    cl_rgb, cl_per = get_opencl_output()

    print("\nResults Summary (Sampled Pixels):")
    for i, p in enumerate([20, 40, 60]):
        print(f"--- {p}% Pixel ---")
        if mojo_gpu_rgb: print(f"Mojo GPU RGB:    {mojo_gpu_rgb[i]}")
        if cl_rgb:       print(f"OpenCL GPU RGB:  {cl_rgb[i]}")
        if mojo_gpu_per: print(f"Mojo GPU Per:    {mojo_gpu_per[i]}")
        if cl_per:       print(f"OpenCL GPU Per:  {cl_per[i]}")

    print("\nValidation Benchmarks (Mojo GPU vs OpenCL):")
    p1 = compare_pixels("RGB Ratio", mojo_gpu_rgb, cl_rgb)
    p2 = compare_pixels("Per Channel", mojo_gpu_per, cl_per)

    if all([p1, p2]):
        print("\n✅ GLOBAL PARITY VALIDATED: Mojo GPU matches OpenCL across sampled pixels.")
    else:
        print("\n❌ PARITY FAILED: Differences detected.")

if __name__ == "__main__":
    main()
