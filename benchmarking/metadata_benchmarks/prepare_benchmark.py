import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import cellxgene_census
from urllib.request import urlretrieve


DATASETS = {
    "human_lung": {
        "id": "066943a2-fdac-4b29-b348-40cede398e4e",  # https://cellxgene.cziscience.com/e/066943a2-fdac-4b29-b348-40cede398e4e.cxg/
        "species": "Homo sapiens",
        "organ": "lung",
    },
    "human_pancreas": {
        "id": "66d15835-5dc8-4e96-b0eb-f48971cb65e8",  # https://cellxgene.cziscience.com/e/66d15835-5dc8-4e96-b0eb-f48971cb65e8.cxg/
        "species": "Homo sapiens",
        "organ": "pancreas",
    },
    "mouse_brain": {
        "id": "3a15ab1c-c36c-4842-9a3e-47e6ffd0ba6f",  # https://cellxgene.cziscience.com/e/3a15ab1c-c36c-4842-9a3e-47e6ffd0ba6f.cxg/
        "species": "Mus musculus",
        "organ": "brain",
    },
    "mouse_kidney": {
        "id": "42bb7f78-cef8-4b0d-9bba-50037d64d8c1",  # https://cellxgene.cziscience.com/e/42bb7f78-cef8-4b0d-9bba-50037d64d8c1.cxg/
        "species": "Mus musculus",
        "organ": "kidney",
    },
    "zebrafish_lens": {
        "id": "https://datasets.cellxgene.cziscience.com/418c08d0-6228-40b2-8160-195ca40a6b77.h5ad",
        "species": "Danio rerio",
        "organ": "lens",
    },
    "zebrafish_blood": {
        "id": "https://datasets.cellxgene.cziscience.com/3becdf3d-31dc-41bd-bbcc-4c21453c51c0.h5ad",
        "species": "Danio rerio",
        "organ": "blood",
    },
}


def _mean_transcript_count(adata) -> float:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    summed = matrix.sum(axis=1)
    return float(np.asarray(summed).ravel().mean())


def _list_census_versions() -> list[str]:
    versions = []
    getter = getattr(cellxgene_census, "get_census_version_directory", None)
    if not getter:
        return versions
    try:
        census_versions = getter()
    except Exception:
        return versions
    if isinstance(census_versions, dict):
        versions = list(census_versions.keys())
    if not versions:
        return versions

    def _sort_key(v: str) -> str:
        if v == "stable":
            return "0"
        if v == "latest":
            return "1"
        return f"2{v}"

    return sorted(versions, key=_sort_key, reverse=True)


def _download_direct_url(url: str, raw_path: Path) -> None:
    urlretrieve(url, raw_path)


def _download_with_probe(
    dataset_name: str,
    dataset_id: str,
    raw_path: Path,
    census_version: str | None,
    probe_versions: bool,
    max_versions: int,
) -> str | None:
    if dataset_id.startswith("http://") or dataset_id.startswith("https://"):
        _download_direct_url(dataset_id, raw_path)
        return None

    if census_version:
        cellxgene_census.download_source_h5ad(
            dataset_id,
            to_path=str(raw_path),
            census_version=census_version,
        )
        return census_version

    try:
        cellxgene_census.download_source_h5ad(dataset_id, to_path=str(raw_path))
        return None
    except KeyError as exc:
        if not probe_versions:
            raise
        if raw_path.exists():
            raw_path.unlink()
        print(f"Dataset id not found in default census version: {dataset_id}")
    except Exception:
        raise

    versions = _list_census_versions()
    if not versions:
        raise KeyError("Unknown dataset_id")

    for version in versions[:max_versions]:
        try:
            print(f"Retrying {dataset_name} with census version: {version}")
            cellxgene_census.download_source_h5ad(
                dataset_id,
                to_path=str(raw_path),
                census_version=version,
            )
            return version
        except KeyError:
            if raw_path.exists():
                raw_path.unlink()
            continue
    raise KeyError("Unknown dataset_id")


def _anonymize_adata(adata):
    blind = adata.copy()
    blind.obs = pd.DataFrame(index=adata.obs.index)
    blind.var = pd.DataFrame(index=adata.var.index)
    blind.obsm.clear()
    blind.obsp.clear()
    blind.varm.clear()
    blind.uns = {}
    if "counts" in blind.layers:
        counts = blind.layers["counts"]
        blind.layers.clear()
        blind.layers["counts"] = counts
    else:
        blind.layers.clear()
    if blind.raw is not None:
        blind.raw = None
    return blind


def prepare_benchmark(
    output_dir: Path,
    skip_download: bool,
    census_version: str | None,
    continue_on_error: bool,
    probe_versions: bool,
    max_versions: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_records = []
    manifest_records = []
    failures = []

    for name, ds_info in DATASETS.items():
        ds_id = ds_info["id"]
        known_species = ds_info["species"]
        known_organ = ds_info.get("organ")
        print(f"--- Processing {name} ({ds_id}) ---")

        raw_path = output_dir / f"{name}_raw.h5ad"
        if not raw_path.exists():
            if skip_download:
                raise FileNotFoundError(f"Missing {raw_path} and --skip-download set.")
            print(f"Downloading {name}...")
            try:
                used_version = _download_with_probe(
                    name,
                    ds_id,
                    raw_path,
                    census_version,
                    probe_versions,
                    max_versions,
                )
            except Exception as exc:
                msg = f"Download failed for {name} ({ds_id}): {exc}"
                print(msg)
                if not continue_on_error:
                    print("Tip: pass --continue-on-error to keep processing other datasets.")
                failures.append({"dataset_name": name, "dataset_id": ds_id, "error": str(exc)})
                if continue_on_error:
                    continue
                raise
        else:
            print(f"File {raw_path} already exists, skipping download.")
            used_version = census_version

        try:
            adata = sc.read_h5ad(raw_path)
        except Exception as exc:
            msg = f"Failed to read {raw_path}: {exc}"
            print(msg)
            failures.append({"dataset_name": name, "dataset_id": ds_id, "error": str(exc)})
            if continue_on_error:
                continue
            raise

        # Species: use curated value from DATASETS; fall back to obs if available.
        if known_species:
            actual_species = known_species
        elif "organism" in adata.obs:
            actual_species = adata.obs["organism"].iloc[0]
        else:
            actual_species = "Unknown"

        # Organ: use curated value from DATASETS; fall back to obs if available.
        if known_organ:
            actual_organ = known_organ
        elif "tissue" in adata.obs:
            actual_organ = adata.obs["tissue"].iloc[0]
        else:
            actual_organ = "Unknown"

        gt_entry = {
            "dataset_name": name,
            "dataset_id": ds_id,
            "census_version": used_version or "",
            "actual_species": actual_species,
            "actual_organ": actual_organ,
            "cell_count": int(adata.n_obs),
            "mean_transcript_count": _mean_transcript_count(adata),
            "raw_h5ad_path": str(raw_path),
        }

        blind = _anonymize_adata(adata)
        blind_path = output_dir / f"{name}_blind.h5ad"
        blind.write_h5ad(blind_path)
        gt_entry["blind_h5ad_path"] = str(blind_path)
        ground_truth_records.append(gt_entry)
        manifest_records.append(
            {
                "dataset_name": name,
                "blind_h5ad_path": str(blind_path),
            }
        )

        print(f"Saved anonymized version to {blind_path}")

    pd.DataFrame(ground_truth_records).to_csv(
        output_dir / "ground_truth_master.csv",
        index=False,
    )
    pd.DataFrame(manifest_records).to_csv(
        output_dir / "benchmark_manifest.csv",
        index=False,
    )
    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "failed_datasets.csv",
            index=False,
        )
    print("\nBenchmark preparation complete.")
    print(f"Ground truth saved to {output_dir / 'ground_truth_master.csv'}")
    print(f"Manifest saved to {output_dir / 'benchmark_manifest.csv'}")
    if failures:
        print(f"Failures saved to {output_dir / 'failed_datasets.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare anonymized CELLxGENE benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "benchmark_data"),
        help="Directory to store raw/anonymized h5ad files and ground truth.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading files if they already exist.",
    )
    parser.add_argument(
        "--census-version",
        default=None,
        help="Optional CELLxGENE Census version string to use for downloads.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other datasets if a download fails.",
    )
    parser.add_argument(
        "--no-probe-versions",
        action="store_false",
        dest="probe_versions",
        help="Disable probing other Census versions when a dataset id is missing.",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        default=3,
        help="Maximum number of Census versions to probe when retrying downloads.",
    )
    parser.set_defaults(probe_versions=True)
    args = parser.parse_args()
    prepare_benchmark(
        Path(os.path.expanduser(args.output_dir)),
        args.skip_download,
        args.census_version,
        args.continue_on_error,
        args.probe_versions,
        args.max_versions,
    )


if __name__ == "__main__":
    main()
