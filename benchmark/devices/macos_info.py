import json
import subprocess


def get_mem_info():
    try:
        mem_info_dict = json.loads(
            subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"])
        )
        mem_info = mem_info_dict["SPHardwareDataType"][0]["physical_memory"]
        return mem_info
    except Exception as e:
        print(f"An error occurred while getting memory info: {e}")
        exit(-1)


def get_gpu_info():
    gpus = []
    try:
        gpu_info_dict = json.loads(
            subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"])
        )
        if "SPDisplaysDataType" in gpu_info_dict:
            gpus = gpu_info_dict["SPDisplaysDataType"]
            print(f"Detected {len(gpus)} GPU(s).")
        else:
            print("No GPU information found.")
    except Exception as e:
        print(f"An error occurred while getting GPU info: {e}")
        exit(-1)

    gpu_info_list = []

    for gpu in gpus:
        info = {}
        info["name"] = gpu.get("sppci_model")
        if gpu.get("spdisplays_vendor") == "sppci_vendor_Apple":
            info["vram"] = f"{get_mem_info()} (shared memory)"
        else:
            info["vram"] = gpu.get("spdisplays_vram")
        info["vendor_id"] = gpu.get("spdisplays_vendor")
        info["cores"] = gpu.get("sppci_cores")
        info["metal"] = gpu.get("spdisplays_mtlgpufamilysupport")
        info["bus"] = gpu.get("sppci_bus")
        info["link"] = gpu.get("spdisplays_pcie_width")
        gpu_info_list.append(info)
    return gpu_info_list
