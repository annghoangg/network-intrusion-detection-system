import os
import pandas as pd


def load_raw_data(input_dir: str = "Input/") -> pd.DataFrame:
    # Dùng pyarrow nếu có để đọc CSV nhanh hơn
    read_kwargs: dict = {}
    try:
        import pyarrow  # noqa: F401
        read_kwargs["engine"] = "pyarrow"
    except ImportError:
        pass

    dfs = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith(".csv"):
                file_path = os.path.join(dirpath, filename)
                print(f"  Loading: {file_path}")
                df_part = pd.read_csv(file_path, **read_kwargs)
                dfs.append(df_part)

    if not dfs:
        raise ValueError(
            f"No CSV files found in '{input_dir}'. "
            "Please check the input directory path."
        )

    print(f"\n  Loaded {len(dfs)} file(s). Concatenating …")
    data = pd.concat(dfs, axis=0, ignore_index=True)

    # Giải phóng bộ nhớ từng file con
    for df_part in dfs:
        del df_part
    del dfs

    print(f"  ✓ Merged dataset shape: {data.shape[0]:,} rows × {data.shape[1]} columns\n")
    return data
