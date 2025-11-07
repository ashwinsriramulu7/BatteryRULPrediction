import scipy.io as sio
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------

def matlab_datevec_to_datetime(vec):
    """Convert MATLAB datevec [Y M D h m s] -> Python datetime."""
    if vec is None:
        return None
    try:
        y, m, d, H, M, S = vec
        return datetime(int(y), int(m), int(d), int(H), int(M), int(S))
    except:
        return None


def safe_get(obj, name, default=None):
    """Return obj.name if exists else default."""
    return getattr(obj, name, default)


def get_any(obj, names, default=None):
    """Try several attribute names in order."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def trunc_to_min_len(fields):
    """Ensure all array lengths match (truncate to shortest length)."""
    lens = []
    for v in fields.values():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            try:
                lens.append(len(v))
            except:
                pass

    if len(lens) == 0:
        return fields

    L = min(lens)

    for k, v in fields.items():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            try:
                fields[k] = v[:L]
            except:
                pass

    return fields

# ----------------------------------------------------------------------
# MAIN NASA PARSER
# ----------------------------------------------------------------------

def extract_cycles(mat_file):
    print(f"\n🔄 Loading {mat_file} ...\n")

    data = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

    # find top-level key like 'B0005'
    key = [k for k in data.keys() if k.startswith("B")][0]
    battery = data[key]

    cycles = np.atleast_1d(battery.cycle)

    charge_rows, discharge_rows, imp_rows = [], [], []

    for idx, cy in enumerate(cycles):

        ctype = safe_get(cy, "type")
        if isinstance(ctype, np.ndarray):
            ctype = ctype.item()
        if not isinstance(ctype, str):
            continue

        d = safe_get(cy, "data")
        if d is None:
            continue

        time_vec = safe_get(d, "Time")
        if time_vec is None:
            continue

        cycle_num = safe_get(cy, "cycle", idx + 1)
        start_time_raw = safe_get(cy, "time")
        ambient = safe_get(cy, "ambient_temperature")

        # convert MATLAB datevec -> datetime
        start_time = matlab_datevec_to_datetime(start_time_raw)

        # universal fields
        voltage = get_any(d, ["Voltage_measured", "Voltage_load", "Voltage_charge"])
        current = get_any(d, ["Current_measured", "Current_load", "Current_charge"])
        temperature = get_any(d, ["Temperature_measured", "Temperature", "T"])
        capacity = safe_get(d, "Capacity")

        # build dictionary
        fields = {
            "time_s": time_vec,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
        }

        # discharge cycles include capacity
        if ctype == "discharge" and capacity is not None:
            fields["capacity_ahr"] = np.array([capacity] * len(time_vec))

        # normalize length
        fields = trunc_to_min_len(fields)
        df = pd.DataFrame(fields)
        N = len(df)

        # attach metadata
        df["cycle_number"] = [cycle_num] * N
        df["start_time"] = [start_time.isoformat() if start_time else None] * N
        df["ambient_temperature"] = [ambient] * N

        # store appropriately
        if ctype == "charge":
            charge_rows.append(df)
        elif ctype == "discharge":
            discharge_rows.append(df)
        elif ctype == "impedance":
            imp_df = pd.DataFrame({
                "cycle_number": [cycle_num],
                "start_time": [start_time.isoformat() if start_time else None],
                "ambient_temperature": [ambient],
                "sense_current": [safe_get(d, "Sense_current")],
                "battery_current": [safe_get(d, "Battery_current")],
                "current_ratio": [safe_get(d, "Current_ratio")],
                "impedance_raw": [safe_get(d, "Battery_impedance")],
                "impedance_rectified": [safe_get(d, "Rectified_impedance")],
                "Re": [safe_get(d, "Re")],
                "Rct": [safe_get(d, "Rct")],
            })
            imp_rows.append(imp_df)

    # finalize
    charge_df = pd.concat(charge_rows, ignore_index=True) if charge_rows else None
    discharge_df = pd.concat(discharge_rows, ignore_index=True) if discharge_rows else None
    imp_df = pd.concat(imp_rows, ignore_index=True) if imp_rows else None

    return charge_df, discharge_df, imp_df

# ----------------------------------------------------------------------
# EXECUTION: Convert B0005.mat → CSV
# ----------------------------------------------------------------------

if __name__ == "__main__":
    mat_file = "Dataset/Raw_Mat_Data/1. BatteryAgingARC-FY08Q4/B0005.mat"

    charge, discharge, imp = extract_cycles(mat_file)

    print("\n✅ Saving CSV files...\n")

    if charge is not None:
        charge.to_csv("B0005_charge.csv", index=False)
        print("✔ Saved B0005_charge.csv")

    if discharge is not None:
        discharge.to_csv("B0005_discharge.csv", index=False)
        print("✔ Saved B0005_discharge.csv")

    if imp is not None:
        imp.to_csv("B0005_impedance.csv", index=False)
        print("✔ Saved B0005_impedance.csv")

    print("\n🎉 Completed successfully! No errors.\n")
