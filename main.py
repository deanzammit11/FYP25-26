from src.combine_seasons import combine_season_files
from src.prepare_features import prepare_features

def main():
    print("Combining season CSV files...")
    combine_season_files()

    print("Preparing features for modeling...")
    prepare_features()

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
