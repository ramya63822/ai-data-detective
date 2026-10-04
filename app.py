"""Command-line smoke test for the AI Data Detective backend."""

from pathlib import Path

from analysis import load_data, profile_dataset
from investigator import investigate_dataset


DATA_FILE = Path("data/sample_sales.csv")


def main() -> None:
    df = load_data(DATA_FILE, DATA_FILE.name)
    profile = profile_dataset(df)

    print("\n=== AI DATA DETECTIVE ===")
    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print(f"Quality score: {profile['quality_score']}/100")

    print("\nRunning investigation with Ollama...")
    report, _, _ = investigate_dataset(df)
    print("\n" + report)


if __name__ == "__main__":
    main()
