"""Comprehensive tests for DataComponentFactory."""

import numpy as np
import pandas as pd
import pytest

from src.data.base import IDataLoader, IDataSplitter, IDataValidator, IScaler
from src.data.factory import (
    DataComponentFactory,
    register_loader,
    register_scaler,
    register_splitter,
    register_validator,
)
from src.data.loaders import CSVDataLoader, IrisDataLoader
from src.data.scalers import MinMaxScaler, StandardScaler
from src.data.splitter import DataSplitter
from src.data.validator import DataValidator


@pytest.fixture
def factory():
    """Create fresh factory instance."""
    return DataComponentFactory()


class TestDataComponentFactoryInitialization:
    """Test factory initialization."""

    def test_factory_initialization(self, factory):
        """Test that factory initializes correctly."""
        assert factory is not None
        assert isinstance(factory, DataComponentFactory)

    def test_defaults_are_registered(self, factory):
        """Test that default components are registered on init."""
        components = factory.list_available_components()

        assert "csv" in components["loaders"]
        assert "iris" in components["loaders"]
        assert "standard" in components["scalers"]
        assert "minmax" in components["scalers"]
        assert "default" in components["validators"]
        assert "default" in components["splitters"]

    def test_defaults_registered_only_once(self):
        """Test that defaults are registered only once across multiple instances."""
        factory1 = DataComponentFactory()
        factory2 = DataComponentFactory()

        assert factory1.list_available_components() == factory2.list_available_components()


class TestDataComponentFactoryLoaders:
    """Test loader creation."""

    def test_create_csv_loader(self, factory):
        """Test creating CSV loader."""
        loader = factory.create_loader("csv")

        assert isinstance(loader, CSVDataLoader)
        assert isinstance(loader, IDataLoader)

    def test_create_iris_loader(self, factory):
        """Test creating Iris loader."""
        loader = factory.create_loader("iris")

        assert isinstance(loader, IrisDataLoader)
        assert isinstance(loader, IDataLoader)

    def test_create_csv_loader_with_kwargs(self, factory):
        """Test creating CSV loader with custom arguments."""
        loader = factory.create_loader("csv", column_names=["a", "b"], encoding="utf-16")

        assert isinstance(loader, CSVDataLoader)
        assert loader._column_names == ["a", "b"]
        assert loader._encoding == "utf-16"

    def test_create_unknown_loader_raises_error(self, factory):
        """Test that creating unknown loader raises error."""
        with pytest.raises(ValueError, match="Unknown loader type"):
            factory.create_loader("unknown_loader")

    def test_create_loader_error_shows_available(self, factory):
        """Test that error message shows available loaders."""
        with pytest.raises(ValueError, match="Available:"):
            factory.create_loader("invalid")


class TestDataComponentFactoryScalers:
    """Test scaler creation."""

    def test_create_standard_scaler(self, factory):
        """Test creating StandardScaler."""
        scaler = factory.create_scaler("standard")

        assert isinstance(scaler, StandardScaler)
        assert isinstance(scaler, IScaler)

    def test_create_minmax_scaler(self, factory):
        """Test creating MinMaxScaler."""
        scaler = factory.create_scaler("minmax")

        assert isinstance(scaler, MinMaxScaler)
        assert isinstance(scaler, IScaler)

    def test_create_scaler_returns_new_instance(self, factory):
        """Test that factory creates new scaler instances (not reusing)."""
        scaler1 = factory.create_scaler("standard")
        scaler2 = factory.create_scaler("standard")

        assert scaler1 is not scaler2
        assert isinstance(scaler1, StandardScaler)
        assert isinstance(scaler2, StandardScaler)

    def test_create_unknown_scaler_raises_error(self, factory):
        """Test that creating unknown scaler raises error."""
        with pytest.raises(ValueError, match="Unknown scaler type"):
            factory.create_scaler("robust_nonexistent")


class TestDataComponentFactoryValidators:
    """Test validator creation."""

    def test_create_default_validator(self, factory):
        """Test creating default validator."""
        validator = factory.create_validator()

        assert isinstance(validator, DataValidator)
        assert isinstance(validator, IDataValidator)

    def test_create_validator_with_explicit_type(self, factory):
        """Test creating validator with explicit type."""
        validator = factory.create_validator("default")

        assert isinstance(validator, DataValidator)

    def test_create_unknown_validator_raises_error(self, factory):
        """Test that creating unknown validator raises error."""
        with pytest.raises(ValueError, match="Unknown validator type"):
            factory.create_validator("strict")


class TestDataComponentFactorySplitters:
    """Test splitter creation."""

    def test_create_default_splitter(self, factory):
        """Test creating default splitter."""
        splitter = factory.create_splitter()

        assert isinstance(splitter, DataSplitter)
        assert isinstance(splitter, IDataSplitter)

    def test_create_splitter_with_explicit_type(self, factory):
        """Test creating splitter with explicit type."""
        splitter = factory.create_splitter("default")

        assert isinstance(splitter, DataSplitter)

    def test_create_unknown_splitter_raises_error(self, factory):
        """Test that creating unknown splitter raises error."""
        with pytest.raises(ValueError, match="Unknown splitter type"):
            factory.create_splitter("stratified")


class TestDataComponentFactoryRegistration:
    """Test component registration."""

    def test_register_custom_loader(self, factory):
        """Test registering custom loader class."""

        class CustomLoader(IDataLoader):
            def load(self, path: str):
                return pd.DataFrame()

        DataComponentFactory.register_loader("custom_test_loader", CustomLoader)

        loader = factory.create_loader("custom_test_loader")
        assert isinstance(loader, CustomLoader)

    def test_register_custom_scaler(self, factory):
        """Test registering custom scaler class."""

        class CustomScaler(IScaler):
            def fit(self, x):
                return self

            def transform(self, x):
                return x

            def fit_transform(self, x):
                return x

        DataComponentFactory.register_scaler("custom_test_scaler", CustomScaler)

        scaler = factory.create_scaler("custom_test_scaler")
        assert isinstance(scaler, CustomScaler)

    def test_register_non_interface_raises_error(self, factory):
        """Test that registering non-interface class raises error."""

        class NotALoader:
            pass

        with pytest.raises(TypeError, match="must implement IDataLoader"):
            DataComponentFactory.register_loader("bad", NotALoader)  # type: ignore[arg-type]

    def test_register_overwrites_with_warning(self, factory):
        """Test that re-registering shows warning but succeeds."""

        class TestOverwriteScaler(IScaler):
            def fit(self, x):
                return self

            def transform(self, x):
                return x

            def fit_transform(self, x):
                return x

        DataComponentFactory.register_scaler("test_overwrite_unique", TestOverwriteScaler)
        DataComponentFactory.register_scaler("test_overwrite_unique", TestOverwriteScaler)

        scaler = factory.create_scaler("test_overwrite_unique")
        assert isinstance(scaler, TestOverwriteScaler)


class TestDataComponentFactoryDecorators:
    """Test decorator-based registration."""

    def test_register_loader_decorator(self, factory):
        """Test @register_loader decorator."""

        @register_loader("json_test")
        class JSONLoader(IDataLoader):
            def load(self, path: str):
                return pd.DataFrame()

        loader = factory.create_loader("json_test")
        assert isinstance(loader, JSONLoader)

    def test_register_scaler_decorator(self, factory):
        """Test @register_scaler decorator."""

        @register_scaler("robust_test")
        class RobustScaler(IScaler):
            def fit(self, x):
                return self

            def transform(self, x):
                return x

            def fit_transform(self, x):
                return x

        scaler = factory.create_scaler("robust_test")
        assert isinstance(scaler, RobustScaler)

    def test_register_validator_decorator(self, factory):
        """Test @register_validator decorator."""

        @register_validator("strict_test")
        class StrictValidator(IDataValidator):
            def validate(self, df):
                return df

            def get_validation_report(self):
                return {}

        validator = factory.create_validator("strict_test")
        assert isinstance(validator, StrictValidator)

    def test_register_splitter_decorator(self, factory):
        """Test @register_splitter decorator."""

        @register_splitter("kfold_test")
        class KFoldSplitter(IDataSplitter):
            def split(
                self, X, y, train_ratio, val_ratio, test_ratio, stratify=True, random_state=42
            ):
                return X, X, X, y, y, y

        splitter = factory.create_splitter("kfold_test")
        assert isinstance(splitter, KFoldSplitter)


class TestDataComponentFactoryUtilities:
    """Test utility methods."""

    def test_list_available_components(self, factory):
        """Test listing available components."""
        components = factory.list_available_components()

        assert "loaders" in components
        assert "scalers" in components
        assert "validators" in components
        assert "splitters" in components

        assert isinstance(components["loaders"], list)
        assert len(components["loaders"]) >= 2
        assert len(components["scalers"]) >= 2

    def test_is_registered_loader(self, factory):
        """Test checking if loader is registered."""
        assert factory.is_registered("loader", "csv") is True
        assert factory.is_registered("loader", "iris") is True
        assert factory.is_registered("loader", "unknown") is False

    def test_is_registered_scaler(self, factory):
        """Test checking if scaler is registered."""
        assert factory.is_registered("scaler", "standard") is True
        assert factory.is_registered("scaler", "minmax") is True
        assert factory.is_registered("scaler", "unknown_scaler_xyz") is False

    def test_is_registered_invalid_type_raises_error(self, factory):
        """Test that checking invalid component type raises error."""
        with pytest.raises(ValueError, match="Invalid component type"):
            factory.is_registered("invalid_type", "anything")

    def test_repr(self, factory):
        """Test string representation."""
        repr_str = repr(factory)

        assert "DataComponentFactory" in repr_str
        assert "total_registered" in repr_str


class TestDataComponentFactoryIntegration:
    """Integration tests with real components."""

    def test_create_and_use_loader(self, factory, tmp_path):
        """Test creating and using a loader."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("1,2,3\n4,5,6\n")

        loader = factory.create_loader("csv", column_names=["a", "b", "c"])
        df = loader.load(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_create_and_use_scaler(self, factory):
        """Test creating and using a scaler."""
        scaler = factory.create_scaler("standard")

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert not np.allclose(X, X_scaled)

    def test_factory_creates_independent_instances(self, factory):
        """Test that factory creates independent scaler instances."""
        scaler1 = factory.create_scaler("standard")
        scaler2 = factory.create_scaler("standard")

        assert isinstance(scaler1, StandardScaler)
        assert isinstance(scaler2, StandardScaler)

        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([[10, 20], [30, 40]])

        scaler1.fit(X1)
        scaler2.fit(X2)

        assert not np.allclose(scaler1.mean_, scaler2.mean_)
