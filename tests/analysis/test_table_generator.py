"""Tests for LaTeX table generator."""

import pandas as pd
import pytest

from src.analysis.table_generator import LaTeXTableGenerator


class TestLaTeXTableGenerator:
    """Test cases for LaTeX table generator."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with experiment results."""
        return pd.DataFrame(
            {
                "test_accuracy": [0.973, 0.969, 0.967, 0.950, 0.945],
                "split_strategy": [
                    "0.8/0.1/0.1",
                    "0.7/0.2/0.1",
                    "0.75/0.15/0.1",
                    "0.8/0.1/0.1",
                    "0.7/0.2/0.1",
                ],
                "scaler_type": ["standard", "minmax", "standard", "standard", "minmax"],
                "layers": ["[64, 32]", "[128, 64]", "[64, 32, 16]", "[32]", "[128]"],
                "activation": ["relu", "tanh", "relu", "relu", "tanh"],
                "learning_rate": [0.01, 0.01, 0.05, 0.001, 0.01],
                "momentum": [0.9, 0.95, 0.9, 0.9, 0.95],
                "batch_size": [16, 32, 8, 16, 32],
                "best_epoch": [48, 52, 45, 67, None],
                "max_epochs": [100, 100, 100, 100, 100],
            }
        )

    @pytest.fixture
    def generator(self):
        """Create LaTeX table generator instance."""
        return LaTeXTableGenerator(top_n=3, sort_by="test_accuracy", ascending=False)

    def test_initialization(self):
        """Test generator initialization with different parameters."""
        gen = LaTeXTableGenerator(top_n=10, sort_by="test_accuracy", ascending=True)

        assert gen.top_n == 10
        assert gen.sort_by == "test_accuracy"
        assert gen.ascending is True

    def test_initialization_defaults(self):
        """Test generator initialization with default parameters."""
        gen = LaTeXTableGenerator()

        assert gen.top_n == 40
        assert gen.sort_by == "test_accuracy"
        assert gen.ascending is False

    def test_generate_table_structure(self, generator, sample_df):
        """Test that generated table has correct LaTeX structure."""
        table = generator.generate(sample_df)

        # Check table environment
        assert "\\begin{table}[H]" in table
        assert "\\end{table}" in table
        assert "\\begin{tabular}" in table
        assert "\\end{tabular}" in table

        # Check table parts
        assert "\\toprule" in table
        assert "\\midrule" in table
        assert "\\bottomrule" in table

        # Check caption and label
        assert "\\caption{" in table
        assert "\\label{tab:best_configs}" in table

    def test_generate_table_headers(self, generator, sample_df):
        """Test that table contains correct column headers in flow order."""
        table = generator.generate(sample_df)

        # All required headers should be present in correct order
        headers = ["No", "Split", "Scal", "Layers", "Act", "LR", "Mom", "Batch", "Epochs", "Acc"]
        for header in headers:
            assert header in table

        # Check header order
        assert "No & Split & Scal & Layers & Act & LR & Mom & Batch & Epochs & Acc" in table

    def test_top_n_selection(self, sample_df):
        """Test that only top N configurations are included."""
        gen = LaTeXTableGenerator(top_n=3)
        table = gen.generate(sample_df)

        # Count data rows (exclude header which also has & and \\)
        lines = table.split("\n")
        data_rows = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert len(data_rows) == 3

    def test_sorting_descending(self, sample_df):
        """Test that results are sorted correctly (descending)."""
        gen = LaTeXTableGenerator(top_n=5, sort_by="test_accuracy", ascending=False)
        table = gen.generate(sample_df)

        # First data row (after midrule) should have highest accuracy (97.3%)
        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert "97.3\\%" in data_lines[0]

    def test_sorting_ascending(self, sample_df):
        """Test that results are sorted correctly (ascending)."""
        gen = LaTeXTableGenerator(top_n=5, sort_by="test_accuracy", ascending=True)
        table = gen.generate(sample_df)

        # First data row should have lowest accuracy (94.5%)
        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert "94.5\\%" in data_lines[0]

    def test_ranking_column(self, generator, sample_df):
        """Test that ranking column starts from 1."""
        table = generator.generate(sample_df)

        # Extract first data row
        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        first_row = data_lines[0]

        # Should start with rank 1
        assert first_row.startswith("1 &")

    def test_accuracy_at_end(self, generator, sample_df):
        """Test that accuracy column is at the end."""
        table = generator.generate(sample_df)

        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        first_row = data_lines[0]

        # Accuracy should be at the end before \\
        assert "97.3\\% \\\\" in first_row

    def test_format_accuracy(self, generator, sample_df):
        """Test accuracy formatting (percentage with 1 decimal, escaped)."""
        table = generator.generate(sample_df)

        # Check for correctly formatted percentages with escaped %
        assert "97.3\\%" in table
        assert "96.9\\%" in table
        assert "96.7\\%" in table

    def test_format_split_strategy(self, generator):
        """Test split strategy formatting."""
        assert generator._format_split("0.8/0.1/0.1") == "80/10/10"
        assert generator._format_split("0.7/0.2/0.1") == "70/20/10"
        assert generator._format_split("0.75/0.15/0.1") == "75/15/10"

    def test_format_scaler(self, generator):
        """Test scaler name formatting."""
        assert generator._format_scaler("standard") == "std"
        assert generator._format_scaler("minmax") == "mm"

    def test_format_layers(self, generator):
        """Test layers formatting (remove spaces)."""
        # Test compact format
        assert generator._format_layers("[64, 32]") == "[64,32]"
        assert generator._format_layers("[128, 64, 32]") == "[128,64,32]"
        assert generator._format_layers("[32]") == "[32]"

    def test_format_learning_rate(self, generator):
        """Test learning rate formatting."""
        # Test different LR values
        assert generator._format_lr(0.01) == ".01"
        assert generator._format_lr(0.05) == ".05"
        assert generator._format_lr(0.001) == ".001"
        assert generator._format_lr(0.1) == ".10"

    def test_format_momentum(self, generator):
        """Test momentum formatting (leading zero stripped)."""
        assert generator._format_momentum(0.9) == ".90"
        assert generator._format_momentum(0.95) == ".95"
        assert generator._format_momentum(0.85) == ".85"

    def test_momentum_in_table(self, generator, sample_df):
        """Test that momentum appears correctly in table."""
        table = generator.generate(sample_df)

        # Check for correctly formatted momentum values (without leading 0)
        assert ".90" in table
        assert ".95" in table
        # Should NOT have leading zero
        assert " 0.90 &" not in table
        assert " 0.95 &" not in table

    def test_format_epochs_with_value(self, generator):
        """Test epochs formatting when best_epoch exists."""
        assert generator._format_epochs(48, 100) == "48/100"
        assert generator._format_epochs(52, 100) == "52/100"
        assert generator._format_epochs(45, 50) == "45/50"

    def test_format_epochs_with_nan(self, generator):
        """Test epochs formatting when best_epoch is NaN."""
        assert generator._format_epochs(None, 100) == "--/100"
        assert generator._format_epochs(pd.NA, 50) == "--/50"

    def test_epochs_in_table(self, generator, sample_df):
        """Test that epochs appear in table as best/total format."""
        table = generator.generate(sample_df)

        # Should contain epoch ranges
        assert "48/100" in table
        assert "52/100" in table
        assert "45/100" in table

    def test_epochs_with_nan_in_table(self, sample_df):
        """Test that NaN epochs show as -- in table."""
        gen = LaTeXTableGenerator(top_n=5, sort_by="test_accuracy", ascending=False)
        table = gen.generate(sample_df)

        # Last row has NaN epoch, should show '--/100'
        assert "--/100" in table

    def test_column_order_flow(self, generator, sample_df):
        """Test that columns follow algorithm flow order."""
        table = generator.generate(sample_df)

        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        first_row = data_lines[0]

        # Split first row by &
        parts = [p.strip() for p in first_row.split("&")]

        # Check order: No, Split, Scal, Layers, Act, LR, Mom, Batch, Epochs, Acc
        assert parts[0].startswith("1")  # No
        assert "/" in parts[1]  # Split (e.g., "80/10/10")
        assert parts[2] in ["std", "mm"]  # Scal
        assert parts[3].startswith("[")  # Layers
        assert parts[4] in ["relu", "tanh"]  # Act
        assert parts[5].startswith(".")  # LR
        assert parts[6].startswith(".")  # Mom
        # parts[7] is Batch (number)
        assert "/" in parts[8]  # Epochs (e.g., "48/100")
        assert "\\%" in parts[9]  # Acc

    def test_save_creates_directory(self, generator, sample_df, tmp_path):
        """Test that save creates parent directories if needed."""
        output_path = tmp_path / "subdir" / "table.tex"
        table = generator.generate(sample_df)

        generator.save(table, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_file_content(self, generator, sample_df, tmp_path):
        """Test that saved file contains correct content."""
        output_path = tmp_path / "table.tex"
        table = generator.generate(sample_df)

        generator.save(table, str(output_path))

        # Read and verify content
        with output_path.open("r") as f:
            saved_content = f.read()

        assert saved_content == table
        assert "\\begin{table}" in saved_content

    def test_complete_row_format(self, generator, sample_df):
        """Test complete row formatting with all columns."""
        table = generator.generate(sample_df)

        # Extract first data row
        lines = table.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        first_row = data_lines[0]

        # Check row structure (10 columns = 9 separators)
        assert first_row.count("&") == 9  # 10 columns = 9 separators
        assert first_row.endswith("\\\\")  # Ends with \\

    def test_repr(self, generator):
        """Test string representation."""
        repr_str = repr(generator)

        assert "LaTeXTableGenerator" in repr_str
        assert "top_n=3" in repr_str
        assert "sort_by='test_accuracy'" in repr_str

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        gen = LaTeXTableGenerator(top_n=10)
        df = pd.DataFrame()

        # Empty DataFrame should raise or be handled gracefully
        # For now, we expect it to fail - this is acceptable behavior
        with pytest.raises(KeyError):
            gen.generate(df)

    def test_fewer_rows_than_top_n(self, sample_df):
        """Test when DataFrame has fewer rows than top_n."""
        gen = LaTeXTableGenerator(top_n=100)  # Request more than available
        table = gen.generate(sample_df)

        # Should include all available rows (5) - but exclude header
        lines = table.split("\n")
        data_rows = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert len(data_rows) == 5

    def test_different_sort_columns(self, sample_df):
        """Test sorting by different columns."""
        # Sort by accuracy (default)
        gen_acc = LaTeXTableGenerator(top_n=3, sort_by="test_accuracy", ascending=False)
        table_acc = gen_acc.generate(sample_df)

        lines = table_acc.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert "97.3\\%" in data_lines[0]  # Highest accuracy first

        # Sort by batch size
        gen_batch = LaTeXTableGenerator(top_n=3, sort_by="batch_size", ascending=True)
        table_batch = gen_batch.generate(sample_df)

        lines = table_batch.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and "&" in line and "\\\\" in line and not line.startswith("No")
        ]
        assert " 8 &" in data_lines[0]  # Smallest batch (8) should be first

    def test_table_caption_and_label(self, generator, sample_df):
        """Test that caption and label are correctly formatted."""
        table = generator.generate(sample_df)

        # Check caption
        assert "\\caption{Najlepsze konfiguracje modelu MLP dla klasyfikacji Iris}" in table

        # Check label
        assert "\\label{tab:best_configs}" in table

    def test_tabular_alignment(self, generator, sample_df):
        """Test that tabular has correct column alignment specification."""
        table = generator.generate(sample_df)

        # Check for correct number of column specifications (10 columns)
        # c l c l c c c c c c = No, Split, Scal, Layers, Act, LR, Mom, Batch, Epochs, Acc
        assert "{c l c l c c c c c c}" in table

    def test_percent_sign_escaped(self, generator, sample_df):
        """Test that percent signs are properly escaped for LaTeX."""
        table = generator.generate(sample_df)

        # Should have escaped percent signs
        assert "\\%" in table
        # Should NOT have unescaped percent signs (except in comments)
        lines = table.split("\n")
        for line in lines:
            if "%" in line and not line.strip().startswith("%"):
                # If line contains %, it must be escaped as \%
                if "%" in line:
                    assert "\\%" in line or line.strip().startswith("\\")
