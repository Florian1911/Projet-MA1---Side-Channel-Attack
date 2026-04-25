import numpy as np
import ctypes
import os
from pathlib import Path

def _ensure_pico_runtime() -> None:
    """Preload PicoSDK runtime libraries before importing picosdk wrappers."""
    candidates = [
        Path("/opt/picoscope/lib/libpicoipp.so"),
        Path("/usr/local/lib/libpicoipp.so"),
        Path("/usr/lib64/libpicoipp.so"),
        Path("/usr/lib/libpicoipp.so"),
    ]
    lib = next((p for p in candidates if p.is_file()), None)
    if lib is None:
        searched = "\n".join(f" - {p}" for p in candidates)
        raise RuntimeError(
            "libpicoipp.so introuvable.\n"
            "Installe PicoSDK, puis ajoute son dossier au runtime linker.\n"
            f"Chemins verifiés:\n{searched}"
        )

    lib_dir = str(lib.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        msg = str(e)
        if "cannot enable executable stack" in msg:
            raise RuntimeError(
                f"Impossible de charger {lib} car elle requiert une pile executable.\n"
                "Corrige le flag GNU_STACK sur la librairie PicoSDK, puis relance:\n"
                "  sudo dnf install -y execstack\n"
                f"  sudo execstack -c {lib}\n"
                "Alternative:\n"
                "  sudo dnf install -y patchelf\n"
                f"  sudo patchelf --clear-execstack {lib}\n"
            ) from e
        raise

_ensure_pico_runtime()

from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok
import matplotlib.pyplot as plt

# PicoSDK may ask to confirm power source after OpenUnit.
PICO_POWER_SUPPLY_NOT_CONNECTED = 286
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282
PROBE_ATTENUATION = 10  # sonde x10

# --- paramètres simples ---
CHANNEL = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]
COUPLING = ps.PS5000A_COUPLING["PS5000A_DC"]
RANGE = ps.PS5000A_RANGE["PS5000A_200MV"]  # ajuste (50mV/100mV/200mV/1V etc.)
OFFSET_V = 0.0

NUM_SAMPLES = 20000
TIMEBASE = 8  # à ajuster; plus petit = plus rapide (selon modèle)
TRIGGER_THRESHOLD_MV = 50  # seuil au bout de la sonde
TRIGGER_DIR = ps.PS5000A_THRESHOLD_DIRECTION["PS5000A_RISING"]

# --- open device ---
chandle = ctypes.c_int16()
status = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, ps.PS5000A_RESOLUTION["PS5000A_DR_12BIT"])
if status in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
    status = ps.ps5000aChangePowerSource(chandle, status)
assert_pico_ok(status)

# --- channel setup ---
status = ps.ps5000aSetChannel(
    chandle,
    CHANNEL,
    1,  # enabled
    COUPLING,
    RANGE,
    ctypes.c_float(OFFSET_V)
)
assert_pico_ok(status)

# --- simple trigger on channel A (optionnel) ---
status = ps.ps5000aSetSimpleTrigger(
    chandle,
    1,  # enabled
    CHANNEL,
    max(1, int(TRIGGER_THRESHOLD_MV / PROBE_ATTENUATION)),  # SDK attend mV a l'entree BNC
    TRIGGER_DIR,
    0,   # delay
    2000 # auto-trigger ms
)
assert_pico_ok(status)

# --- query time interval ---
time_interval_ns = ctypes.c_float()
max_samples = ctypes.c_int32()
status = ps.ps5000aGetTimebase2(
    chandle, TIMEBASE, NUM_SAMPLES,
    ctypes.byref(time_interval_ns),
    0,
    ctypes.byref(max_samples),
    0
)
assert_pico_ok(status)

# --- run block ---
ready = ctypes.c_int16(0)
status = ps.ps5000aRunBlock(chandle, 0, NUM_SAMPLES, TIMEBASE, None, 0, None, None)
assert_pico_ok(status)

while not ready.value:
    status = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    assert_pico_ok(status)

# --- buffers ---
buffer_max = (ctypes.c_int16 * NUM_SAMPLES)()
buffer_min = (ctypes.c_int16 * NUM_SAMPLES)()
status = ps.ps5000aSetDataBuffers(
    chandle, CHANNEL,
    ctypes.byref(buffer_max),
    ctypes.byref(buffer_min),
    NUM_SAMPLES,
    0,
    ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"]
)
assert_pico_ok(status)

# --- get values ---
num = ctypes.c_int32(NUM_SAMPLES)
overflow = ctypes.c_int16()
status = ps.ps5000aGetValues(
    chandle, 0, ctypes.byref(num), 1,
    ps.PS5000A_RATIO_MODE["PS5000A_RATIO_MODE_NONE"],
    0, ctypes.byref(overflow)
)
assert_pico_ok(status)

# --- convert ADC -> mV ---
max_adc = ctypes.c_int16()
status = ps.ps5000aMaximumValue(chandle, ctypes.byref(max_adc))
assert_pico_ok(status)

mv = np.array(buffer_max[:num.value], dtype=np.int16)
mv = ps.adc2mV(mv, RANGE, max_adc)
mv = mv * PROBE_ATTENUATION

t = np.arange(num.value) * (time_interval_ns.value * 1e-9)

np.save("trace_mv.npy", mv.astype(np.float32))
np.save("time_s.npy", t.astype(np.float64))

print(
    f"Saved trace_mv.npy ({num.value} samples), dt={time_interval_ns.value} ns, "
    f"overflow={overflow.value}, probe=x{PROBE_ATTENUATION}"
)

plt.plot(t, mv)
plt.xlabel("Time (s)")
plt.ylabel("mV")
plt.show()

ps.ps5000aStop(chandle)
ps.ps5000aCloseUnit(chandle)
