import pandas as pd
import numpy as np


class PythonAgent:

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load CSV or Excel dataset."""

        if file_path.lower().endswith(".csv"):
            return pd.read_csv(file_path)

        if file_path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path)

        raise ValueError("Unsupported file type")

    def get_basic_info(self, df: pd.DataFrame) -> dict:
        """Get basic dataset information."""

        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": [str(column) for column in df.columns],
        }

    def get_missing_values(self, df: pd.DataFrame) -> dict:
        """Get missing values."""

        missing = df.isna().sum()

        return {
            str(column): int(count)
            for column, count in missing.items()
        }

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get statistics for numerical columns."""

        numerical_df = df.select_dtypes(include=np.number)

        if numerical_df.empty:
            return {}

        statistics = {}

        for column in numerical_df.columns:

            statistics[str(column)] = {
                "mean": float(numerical_df[column].mean()),
                "min": float(numerical_df[column].min()),
                "max": float(numerical_df[column].max()),
                "std": float(numerical_df[column].std()),
            }

        return statistics

    def answer_question(
        self,
        df: pd.DataFrame,
        question: str
    ) -> str:
        """Answer simple natural-language questions."""

        question = question.lower().strip()

        # -----------------------------------------
        # Number of rows
        # -----------------------------------------
        if "how many rows" in question or "number of rows" in question:
            return f"The dataset contains {len(df)} rows."

        # -----------------------------------------
        # Number of columns
        # -----------------------------------------
        if "how many columns" in question or "number of columns" in question:
            return f"The dataset contains {len(df.columns)} columns."

        # -----------------------------------------
        # Column names
        # -----------------------------------------
        if (
            "column names" in question
            or "what are the columns" in question
            or "list columns" in question
        ):
            columns = ", ".join(str(column) for column in df.columns)
            return f"The columns are: {columns}"

        # -----------------------------------------
        # Missing values
        # -----------------------------------------
        if "missing" in question:
            missing = df.isna().sum()
            result = []

            for column, count in missing.items():
                if count > 0:
                    result.append(f"{column}: {count}")

            if not result:
                return "There are no missing values in the dataset."

            return "Missing values: " + ", ".join(result)

        # -----------------------------------------
        # Find average / mean
        # -----------------------------------------
        if "average" in question or "mean" in question:

            numerical_columns = df.select_dtypes(
                include=np.number
            ).columns

            for column in numerical_columns:

                if str(column).lower() in question:
                    value = df[column].mean()

                    return (
                        f"The average of {column} is "
                        f"{value:.2f}."
                    )

            return (
                "Please mention a numerical column "
                "for which you want the average."
            )

        # -----------------------------------------
        # Find maximum
        # -----------------------------------------
        if (
            "maximum" in question
            or "max" in question
            or "highest" in question
        ):

            numerical_columns = df.select_dtypes(
                include=np.number
            ).columns

            for column in numerical_columns:

                if str(column).lower() in question:
                    value = df[column].max()

                    return (
                        f"The maximum {column} is "
                        f"{value}."
                    )

            return (
                "Please mention a numerical column "
                "for which you want the maximum."
            )

        # -----------------------------------------
        # Find minimum
        # -----------------------------------------
        if (
            "minimum" in question
            or "min" in question
            or "lowest" in question
        ):

            numerical_columns = df.select_dtypes(
                include=np.number
            ).columns

            for column in numerical_columns:

                if str(column).lower() in question:
                    value = df[column].min()

                    return (
                        f"The minimum {column} is "
                        f"{value}."
                    )

            return (
                "Please mention a numerical column "
                "for which you want the minimum."
            )

        # -----------------------------------------
        # Default response
        # -----------------------------------------
        return (
            "I understand that you want to analyze the dataset, "
            "but I could not identify the exact operation. "
            "Try asking about the average, maximum, minimum, "
            "missing values, rows, or columns."
        )

    def analyze(self, file_path: str) -> dict:
        """Perform complete dataset analysis."""

        df = self.load_dataset(file_path)

        return {
            "basic_info": self.get_basic_info(df),
            "missing_values": self.get_missing_values(df),
            "statistics": self.get_statistics(df),
        }