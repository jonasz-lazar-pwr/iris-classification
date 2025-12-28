from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import CSVDataLoader, IrisDataLoader


class TestCSVDataLoader:
    """Test suite for CSVDataLoader."""

    def test_init_with_column_names(self):
        """Test initialization with column names."""
        columns = ["col1", "col2", "col3"]
        loader = CSVDataLoader(column_names=columns)

        assert loader._column_names == columns

    def test_init_without_column_names(self):
        """Test initialization without column names."""
        loader = CSVDataLoader()

        assert loader._column_names is None

    def test_load_success(self, tmp_path: Path):
        """Test successful CSV loading."""
        csv_file = tmp_path / "test.csv"
        csv_content = "1,2,3\n4,5,6\n7,8,9\n"
        csv_file.write_text(csv_content)

        loader = CSVDataLoader(column_names=["a", "b", "c"])
        df = loader.load(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert len(df.columns) == 3
        assert list(df.columns) == ["a", "b", "c"]
        assert df["a"].tolist() == [1, 4, 7]

    def test_load_without_column_names(self, tmp_path: Path):
        """Test loading CSV without specifying column names."""
        csv_file = tmp_path / "test.csv"
        csv_content = "1,2,3\n4,5,6\n"
        csv_file.write_text(csv_content)

        loader = CSVDataLoader()
        df = loader.load(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [0, 1, 2]

    def test_load_file_not_found(self):
        """Test loading non-existent file raises FileNotFoundError."""
        loader = CSVDataLoader()

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            loader.load("non_existent_file.csv")

    def test_load_invalid_csv(self, tmp_path: Path):
        """Test loading invalid CSV raises ParserError."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("1,2,3\n4,5\n6,7,8,9,10\n")

        loader = CSVDataLoader()

        with pytest.raises(pd.errors.ParserError):
            loader.load(str(csv_file))

    def test_load_empty_csv(self, tmp_path: Path):
        """Test loading empty CSV file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        loader = CSVDataLoader(column_names=["a", "b"])
        df = loader.load(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestIrisDataLoader:
    """Test suite for IrisDataLoader."""

    def test_init(self):
        """Test IrisDataLoader initialization."""
        loader = IrisDataLoader()

        assert loader._column_names == IrisDataLoader.COLUMN_NAMES
        assert len(loader._column_names) == 5

    def test_column_names_constant(self):
        """Test COLUMN_NAMES constant is correct."""
        expected = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

        assert IrisDataLoader.COLUMN_NAMES == expected

    def test_species_map_constant(self):
        """Test SPECIES_MAP constant is correct."""
        expected = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2,
        }

        assert IrisDataLoader.SPECIES_MAP == expected

    def test_load_success(self, tmp_path: Path):
        """Test successful Iris dataset loading."""
        csv_file = tmp_path / "iris.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
            "4.9,3.0,1.4,0.2,Iris-setosa\n"
        )
        csv_file.write_text(csv_content)

        loader = IrisDataLoader()
        df = loader.load(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert "species_encoded" in df.columns
        assert df["species_encoded"].tolist() == [0, 1, 2, 0]
        # ✅ ZMIANA: species is now int
        assert df["species"].tolist() == [0, 1, 2, 0]

    def test_load_species_encoding(self, tmp_path: Path):
        """Test species label encoding is correct."""
        csv_file = tmp_path / "iris.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
        )
        csv_file.write_text(csv_content)

        loader = IrisDataLoader()
        df = loader.load(str(csv_file))

        # ✅ ZMIANA: species is now int, so check directly
        assert df["species"].iloc[0] == 0  # Iris-setosa
        assert df["species"].iloc[1] == 1  # Iris-versicolor
        assert df["species"].iloc[2] == 2  # Iris-virginica

        # Also verify species_encoded
        assert df["species_encoded"].iloc[0] == 0
        assert df["species_encoded"].iloc[1] == 1
        assert df["species_encoded"].iloc[2] == 2

    def test_load_with_unknown_species(self, tmp_path: Path, caplog):
        """Test loading with unknown species removes invalid rows."""
        csv_file = tmp_path / "iris_bad.csv"
        csv_content = (
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-unknown\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
        )
        csv_file.write_text(csv_content)

        loader = IrisDataLoader()
        df = loader.load(str(csv_file))

        assert "unknown species labels" in caplog.text.lower()

        assert len(df) == 2
        # ✅ ZMIANA: species is now int (0, 2), not strings
        assert df["species"].tolist() == [0, 2]  # setosa=0, virginica=2

        # ✅ ZMIANA: Check that only valid encoded values exist
        assert all(df["species"].isin([0, 2]))
        assert df["species_encoded"].isnull().sum() == 0

    def test_load_real_iris_dataset(self, iris_data_path: Path):
        """Test loading the real Iris dataset."""
        if not iris_data_path.exists():
            pytest.skip("Real Iris dataset not found")

        loader = IrisDataLoader()
        df = loader.load(str(iris_data_path))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert "species_encoded" in df.columns
        assert df["species_encoded"].notna().all()

        # ✅ ZMIANA: species is now int (0, 1, 2)
        distribution = df["species"].value_counts()
        assert distribution[0] == 50  # Iris-setosa
        assert distribution[1] == 50  # Iris-versicolor
        assert distribution[2] == 50  # Iris-virginica

    def test_load_inherits_parent_validation(self, tmp_path: Path):
        """Test that IrisDataLoader inherits file validation from parent."""
        loader = IrisDataLoader()

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            loader.load("non_existent_iris.csv")

    def test_load_all_columns_present(self, tmp_path: Path):
        """Test all expected columns are present after loading."""
        csv_file = tmp_path / "iris.csv"
        csv_content = "5.1,3.5,1.4,0.2,Iris-setosa\n"
        csv_file.write_text(csv_content)

        loader = IrisDataLoader()
        df = loader.load(str(csv_file))

        expected_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
            "species_encoded",
        ]

        for col in expected_columns:
            assert col in df.columns


class TestLoadersIntegration:
    """Integration tests for loaders."""

    def test_csv_to_iris_loader_compatibility(self, tmp_path: Path):
        """Test that IrisDataLoader is compatible with CSVDataLoader interface."""
        csv_file = tmp_path / "test.csv"
        csv_content = "5.1,3.5,1.4,0.2,Iris-setosa\n"
        csv_file.write_text(csv_content)

        csv_loader = CSVDataLoader(column_names=IrisDataLoader.COLUMN_NAMES)
        iris_loader = IrisDataLoader()

        df_csv = csv_loader.load(str(csv_file))
        df_iris = iris_loader.load(str(csv_file))

        assert "species_encoded" not in df_csv.columns
        assert "species_encoded" in df_iris.columns
        # ✅ ZMIANA: IrisDataLoader has same base columns + species_encoded
        assert len(df_iris.columns) == len(df_csv.columns) + 1

    @pytest.mark.parametrize(
        "species,expected_code",
        [
            ("Iris-setosa", 0),
            ("Iris-versicolor", 1),
            ("Iris-virginica", 2),
        ],
    )
    def test_species_encoding_parametrized(self, tmp_path: Path, species: str, expected_code: int):
        """Test encoding for each species type."""
        csv_file = tmp_path / "iris.csv"
        csv_content = f"5.1,3.5,1.4,0.2,{species}\n"
        csv_file.write_text(csv_content)

        loader = IrisDataLoader()
        df = loader.load(str(csv_file))

        # ✅ ZMIANA: Check both species and species_encoded
        assert df["species"].iloc[0] == expected_code
        assert df["species_encoded"].iloc[0] == expected_code
