"""
Device Manager for OpenVINO Benchmark Studio

Detects available OpenVINO devices using the Python API
"""

from typing import List, Dict


class DeviceManager:
    """Manages OpenVINO device detection and information"""

    def __init__(self):
        self.available_devices = []

    def detect_devices(self) -> List[Dict[str, str]]:
        """Detect available OpenVINO devices with capabilities"""
        devices = []

        try:
            import openvino as ov
            core = ov.Core()

            for device_name in core.available_devices:
                try:
                    full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
                    try:
                        caps = list(core.get_property(device_name, "OPTIMIZATION_CAPABILITIES"))
                    except Exception:
                        caps = []
                    try:
                        device_type = str(core.get_property(device_name, "DEVICE_TYPE"))
                    except Exception:
                        device_type = "integrated" if device_name == "CPU" else "unknown"

                    devices.append({
                        'name': device_name,
                        'full_name': full_name,
                        'type': device_type,
                        'capabilities': caps,
                    })
                except Exception as e:
                    print(f"Warning: Could not get properties for {device_name}: {e}")
                    devices.append({
                        'name': device_name,
                        'full_name': device_name,
                        'type': 'unknown',
                        'capabilities': [],
                    })

        except Exception as e:
            print(f"Warning: Error detecting devices: {e}. Using default.")
            devices = [{'name': 'CPU', 'full_name': 'CPU (Default)', 'type': 'integrated', 'capabilities': []}]

        self.available_devices = devices
        return devices

    def is_device_available(self, device_name: str) -> bool:
        return any(d['name'] == device_name for d in self.available_devices)

    def get_device_info(self, device_name: str) -> Dict[str, str]:
        for device in self.available_devices:
            if device['name'] == device_name:
                return device
        return None
