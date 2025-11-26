import pandas as pd
import sys
import os

# Hardcoded file paths
INPUT_FILE = "../datasets/Cleaned/clean_recipes.csv"
OUTPUT_FILE_BASE = "../datasets/Cleaned/clean_recipes"

def main():
	try:
		n_rows = int(input('Enter number of top rows to extract: '))
	except ValueError:
		print('Invalid number for rows.')
		sys.exit(1)

	if not os.path.isfile(INPUT_FILE):
		print(f"Input file '{INPUT_FILE}' does not exist.")
		sys.exit(1)

	output_file = f"{OUTPUT_FILE_BASE}_{n_rows}.csv"

	try:
		df = pd.read_csv(INPUT_FILE)
		df_head = df.head(n_rows)
		df_head.to_csv(output_file, index=False)
		print(f"Saved top {n_rows} rows to '{output_file}'")
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)

if __name__ == "__main__":
	main()
