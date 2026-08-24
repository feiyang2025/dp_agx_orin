import os
import subprocess
import threading
import time
from pathlib import Path

from cereal import log
from openpilot.system.hardware.base import HardwareBase, LPABase, ThermalConfig, ThermalZone

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class Jetson(HardwareBase):
  def __init__(self):
    self.dfs = None
    self._watchdog_stop = threading.Event()
    self._ina_hwmon_dirs: list[str] | None = None  # lazy-discovered INA3221 hwmon paths

  @staticmethod
  def _find_ina_hwmon_dirs() -> list[str]:
    """Discover INA3221 power monitor hwmon dirs.

    Path differs per board: AGX Xavier = i2c bus 1 (1-0040),
    AGX Orin / TW-T906G = bus 3 or c (3-0040 / c-0040), driver ina3221 or ina3221x.
    Scan instead of hardcoding so one build works on both boards.
    """
    import glob
    dirs: list[str] = []
    for pattern in ["/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*",
                    "/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*"]:
      dirs.extend(glob.glob(pattern))
    return sorted(set(dirs))

  def _ina_paths(self) -> list[str]:
    if self._ina_hwmon_dirs is None:
      self._ina_hwmon_dirs = self._find_ina_hwmon_dirs()
    return self._ina_hwmon_dirs

  def get_os_version(self):
    try:
      with open("/etc/nv_tegra_release") as f:
        return f.read().strip().split(",")[0]
    except FileNotFoundError:
      return "JetPack unknown"

  def get_device_type(self):
    return "pc"  # capnp DeviceType has no "jetson"; use "pc" as closest match

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def uninstall(self):
    pass

  def get_imei(self, slot):
    return ""

  def get_serial(self):
    for path in ["/sys/module/fuse_burn/parameters/tegra_chip_uid",
                 "/sys/module/tegra_fuse/parameters/tegra_chip_uid"]:
      try:
        with open(path) as f:
          return f.read().strip()
      except FileNotFoundError:
        continue
    return "jetson-unknown"

  def get_network_info(self):
    return None

  def get_network_type(self):
    # Check for ethernet first, then wifi
    try:
      for iface in os.listdir("/sys/class/net/"):
        if iface == "lo":
          continue
        operstate_path = f"/sys/class/net/{iface}/operstate"
        if os.path.exists(operstate_path):
          with open(operstate_path) as f:
            if f.read().strip() == "up":
              if iface.startswith("eth") or iface.startswith("enp"):
                return NetworkType.ethernet
              elif iface.startswith("wlan") or iface.startswith("wlp"):
                return NetworkType.wifi
    except Exception:
      pass
    return NetworkType.none

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_sim_lpa(self) -> LPABase:
    raise NotImplementedError("SIM LPA not available on Jetson")

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_current_power_draw(self):
    # INA3221 power monitor (channels: GPU/CPU/SoC on Xavier; board-specific on Orin)
    total_power = 0
    for hwmon in self._ina_paths():
      for channel in range(1, 4):
        try:
          with open(f"{hwmon}/in{channel}_input") as f:
            voltage_mv = int(f.read().strip())
          with open(f"{hwmon}/curr{channel}_input") as f:
            current_ma = int(f.read().strip())
          total_power += (voltage_mv * current_ma) / 1_000_000  # Convert to watts
        except (FileNotFoundError, ValueError, PermissionError):
          continue
    return total_power

  def get_som_power_draw(self):
    return self.get_current_power_draw()

  def shutdown(self):
    subprocess.check_output(["sudo", "shutdown", "-h", "now"])

  def get_thermal_config(self):
    # Zone names differ between Xavier and Orin; probe sysfs at runtime.
    # Xavier: CPU-therm / GPU-therm / Tdiode_tegra / PMIC-Die
    # Orin candidates: CV0-therm, TdiodeTEGRA, etc.
    available = {}
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
      try:
        available[(z / "type").read_text().strip()] = z.name
      except OSError:
        continue

    def zone(*candidates: str) -> str | None:
      for c in candidates:
        if c in available:
          return c
      return None

    cpu = zone("CPU-therm", "CV0-therm")
    gpu = zone("GPU-therm", "CV1-therm", "GPU-therm")
    memory = zone("Tdiode_tegra", "TdiodeTEGRA", "CV2-therm")
    pmic = zone("PMIC-Die", "PMIC_thermal")

    return ThermalConfig(
      cpu=[ThermalZone(cpu)] if cpu else [],
      gpu=[ThermalZone(gpu)] if gpu else [],
      memory=ThermalZone(memory) if memory else None,
      pmic=[ThermalZone(pmic)] if pmic else [],
    )

  def set_screen_brightness(self, percentage):
    pass

  def get_screen_brightness(self):
    return 0

  def set_power_save(self, powersave_enabled):
    # nvpmodel: 0 = MAXN (30W), 2 = MODE_15W, 1 = MODE_10W
    try:
      if powersave_enabled:
        subprocess.check_output(["sudo", "nvpmodel", "-m", "1"], stderr=subprocess.STDOUT)  # 10W parked
        # Disable cores 4-7 to save power when parked
        for core in range(4, 8):
          try:
            with open(f"/sys/devices/system/cpu/cpu{core}/online", 'w') as f:
              f.write("0")
          except (OSError, PermissionError):
            pass
      else:
        # Re-enable all cores
        for core in range(4, 8):
          try:
            with open(f"/sys/devices/system/cpu/cpu{core}/online", 'w') as f:
              f.write("1")
          except (OSError, PermissionError):
            pass
        subprocess.check_output(["sudo", "nvpmodel", "-m", "0"], stderr=subprocess.STDOUT)  # MAXN driving
        subprocess.check_output(["sudo", "jetson_clocks"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
      pass

  def get_gpu_usage_percent(self):
    try:
      with open("/sys/devices/gpu.0/load") as f:
        return int(f.read().strip()) / 10
    except (FileNotFoundError, ValueError):
      return 0

  def get_modem_temperatures(self):
    return []

  def initialize_hardware(self):
    # Lock clocks to maximum for consistent performance
    try:
      subprocess.check_output(["sudo", "nvpmodel", "-m", "0"], stderr=subprocess.STDOUT)  # MAXN 30W
      subprocess.check_output(["sudo", "jetson_clocks"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
      pass

    # Start watchdog thread to reapply clocks if thermal throttling undoes them
    threading.Thread(target=self._jetson_clocks_watchdog, daemon=True).start()

    # Initialize DFS for dynamic power management
    try:
      from openpilot.system.hardware.jetson.dfs import JetsonDFS
      self.dfs = JetsonDFS()
      self.dfs.initialize()
    except Exception:
      self.dfs = None

    # Initialize huge pages for CUDA memory (5-10% TLB improvement)
    try:
      from openpilot.system.hardware.jetson.hugepages import initialize as init_hugepages
      init_hugepages()
    except Exception:
      pass

  def _jetson_clocks_watchdog(self):
    """Re-apply jetson_clocks if thermal throttling undoes frequency lock."""
    while not self._watchdog_stop.is_set():
      self._watchdog_stop.wait(timeout=120)  # Check every 2 minutes, interruptible
      if self._watchdog_stop.is_set():
        break
      try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
          freq = int(f.read().strip())
        # Throttle detection relative to max freq (Xavier 2.2GHz, Orin 2.2GHz; board-agnostic)
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") as f:
          max_freq = int(f.read().strip())
        if max_freq > 0 and freq < max_freq * 0.9:  # >10% below max means throttled
          subprocess.call(["sudo", "jetson_clocks"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
      except Exception:
        pass

  def stop_watchdog(self):
    """Stop the watchdog thread cleanly."""
    self._watchdog_stop.set()

  def get_voltage(self) -> float:
    """Read bus voltage in millivolts from INA3221 power monitor (first discovered hwmon)."""
    for hwmon in self._ina_paths():
      for channel in range(1, 4):
        try:
          with open(f"{hwmon}/in{channel}_input") as f:
            return float(f.read().strip())  # millivolts
        except (OSError, ValueError):
          continue
    return 0.0

  def get_current(self) -> float:
    """Read total bus current in milliamps from INA3221 power monitor."""
    total_ma = 0.0
    for hwmon in self._ina_paths():
      for channel in range(1, 4):
        try:
          with open(f"{hwmon}/curr{channel}_input") as f:
            total_ma += float(f.read().strip())
        except (OSError, ValueError):
          continue
    return total_ma

  def get_networks(self):
    return None
