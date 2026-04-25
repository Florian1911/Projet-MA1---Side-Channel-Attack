import ctypes
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
from picosdk.functions import assert_pico_ok, adc2mV

# PicoSDK may ask to confirm power source after OpenUnit.
PICO_POWER_SUPPLY_NOT_CONNECTED = 286
PICO_USB3_0_DEVICE_NON_USB3_0_PORT = 282

chandle = ctypes.c_int16()
status = {}

# ouvrir le scope
resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"]

status["openunit"] = ps.ps5000aOpenUnit(
    ctypes.byref(chandle),
    None,
    resolution
)
if status["openunit"] in (PICO_POWER_SUPPLY_NOT_CONNECTED, PICO_USB3_0_DEVICE_NON_USB3_0_PORT):
    status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, status["openunit"])
    assert_pico_ok(status["changePowerSource"])
else:
    assert_pico_ok(status["openunit"])

print("PicoScope connecté")

# config canal A
channel = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_A"]

status["setChA"] = ps.ps5000aSetChannel(
    chandle,
    channel,
    1,
    ps.PS5000A_COUPLING["PS5000A_DC"],
    ps.PS5000A_RANGE["PS5000A_500MV"],
    0
)
assert_pico_ok(status["setChA"])

# nombre d'échantillons
preTriggerSamples = 0
postTriggerSamples = 200000
maxSamples = preTriggerSamples + postTriggerSamples

timebase = 100
timeIntervalns = ctypes.c_float()
returnedMaxSamples = ctypes.c_int32()

status["getTimebase2"] = ps.ps5000aGetTimebase2(
    chandle,
    timebase,
    maxSamples,
    ctypes.byref(timeIntervalns),
    ctypes.byref(returnedMaxSamples),
    0
)
assert_pico_ok(status["getTimebase2"])

# lancer acquisition
status["runBlock"] = ps.ps5000aRunBlock(
    chandle,
    preTriggerSamples,
    postTriggerSamples,
    timebase,
    None,
    0,
    None,
    None
)
assert_pico_ok(status["runBlock"])

# attendre la fin
ready = ctypes.c_int16(0)

while ready.value == 0:
    status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))

# buffers
bufferA = (ctypes.c_int16 * maxSamples)()

status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(
    chandle,
    channel,
    ctypes.byref(bufferA),
    None,
    maxSamples,
    0,
    0
)
assert_pico_ok(status["setDataBuffersA"])

cmaxSamples = ctypes.c_int32(maxSamples)

status["getValues"] = ps.ps5000aGetValues(
    chandle,
    0,
    ctypes.byref(cmaxSamples),
    0,
    0,
    0,
    None
)
assert_pico_ok(status["getValues"])

# convertir ADC -> mV
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

adc = np.array(bufferA[:cmaxSamples.value])
mv = adc2mV(adc, ps.PS5000A_RANGE["PS5000A_500MV"], maxADC)

plt.plot(mv)
plt.title("Trace PicoScope")
plt.xlabel("Samples")
plt.ylabel("mV")
plt.show()

ps.ps5000aCloseUnit(chandle)
