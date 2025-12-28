"""Comprehensive tests for Trainer with composition pattern."""

import numpy as np
import pytest

from src.models.activations import ReLU, Softmax
from src.models.losses import CrossEntropyLoss
from src.models.mlp import MLP
from src.models.optimizers import SGDMomentum
from src.training.trainer import Trainer


@pytest.fixture
def simple_model():
    """Create simple MLP model for testing."""
    return MLP(
        layer_sizes=[4, 8, 3],
        activations=[ReLU(), Softmax()],
        loss_function=CrossEntropyLoss(),
        optimizer=SGDMomentum(learning_rate=0.01),
        random_state=42,
    )


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X_train = np.random.randn(20, 4)
    y_train_indices = np.random.randint(0, 3, size=20)
    y_train = np.zeros((20, 3))
    y_train[np.arange(20), y_train_indices] = 1

    X_val = np.random.randn(10, 4)
    y_val_indices = np.random.randint(0, 3, size=10)
    y_val = np.zeros((10, 3))
    y_val[np.arange(10), y_val_indices] = 1

    return X_train, y_train, X_val, y_val


class TestTrainerInitialization:
    """Test trainer initialization."""

    def test_trainer_initialization(self, simple_model):
        """Test that trainer initializes correctly."""
        trainer = Trainer(model=simple_model, epochs=10, batch_size=5)

        assert trainer.model is simple_model
        assert trainer.epochs == 10
        assert trainer.batch_size == 5
        assert trainer.patience == 10
        assert trainer.history is None

    def test_trainer_with_custom_parameters(self, simple_model):
        """Test trainer with custom parameters."""
        trainer = Trainer(
            model=simple_model,
            epochs=50,
            batch_size=16,
            patience=20,
            random_state=123,
        )

        assert trainer.epochs == 50
        assert trainer.batch_size == 16
        assert trainer.patience == 20
        assert trainer.random_state == 123

    def test_trainer_default_parameters(self, simple_model):
        """Test trainer with default parameters."""
        trainer = Trainer(model=simple_model)

        assert trainer.epochs == 100
        assert trainer.batch_size == 32
        assert trainer.patience == 10
        assert trainer.random_state is None

    def test_trainer_with_disabled_early_stopping(self, simple_model):
        """Test trainer with early stopping disabled."""
        trainer = Trainer(model=simple_model, patience=None)  # ← NOWY TEST

        assert trainer.patience is None


class TestTrainerFit:
    """Test trainer fit method."""

    def test_fit_without_validation(self, simple_model, sample_data):
        """Test training without validation data."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        history = trainer.fit(X_train, y_train)

        assert "train_loss" in history
        assert "train_accuracy" in history
        assert "val_loss" not in history
        assert "val_accuracy" not in history
        assert len(history["train_loss"]) == 5
        assert len(history["train_accuracy"]) == 5
        # ← NOWY: Check metadata
        assert "best_epoch" in history
        assert "stopped_early" in history
        assert history["stopped_early"] is False  # No validation = no early stopping

    def test_fit_with_validation(self, simple_model, sample_data):
        """Test training with validation data."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=10)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        assert "train_loss" in history
        assert "train_accuracy" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        # ← NOWY: Check early stopping metadata
        assert "best_epoch" in history
        assert "best_val_accuracy" in history
        assert "stopped_early" in history
        assert "total_epochs_trained" in history

    def test_fit_updates_model_weights(self, simple_model, sample_data):
        """Test that training updates model weights."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=10, batch_size=5, patience=None)

        initial_params = simple_model.get_parameters()
        trainer.fit(X_train, y_train)
        trained_params = simple_model.get_parameters()

        assert not np.array_equal(initial_params["layer0_W"], trained_params["layer0_W"])

    def test_fit_improves_accuracy(self, simple_model, sample_data):
        """Test that training improves accuracy."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=50, batch_size=5, patience=None)

        history = trainer.fit(X_train, y_train)

        initial_accuracy = history["train_accuracy"][0]
        final_accuracy = history["train_accuracy"][-1]

        assert final_accuracy >= initial_accuracy

    def test_fit_stores_history(self, simple_model, sample_data):
        """Test that fit stores history in trainer."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        assert trainer.history is None

        history = trainer.fit(X_train, y_train)

        assert trainer.history is history
        assert trainer.get_history() is history

    def test_fit_mismatched_train_shapes_raises_error(self, simple_model):
        """Test that mismatched train shapes raise error."""
        X_train = np.random.randn(20, 4)
        y_train = np.random.randn(15, 3)

        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        with pytest.raises(ValueError, match="must have same number of samples"):
            trainer.fit(X_train, y_train)

    def test_fit_mismatched_val_shapes_raises_error(self, simple_model, sample_data):
        """Test that mismatched validation shapes raise error."""
        X_train, y_train, _, _ = sample_data
        X_val = np.random.randn(10, 4)
        y_val = np.random.randn(5, 3)

        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=10)

        with pytest.raises(ValueError, match="must have same number of samples"):
            trainer.fit(X_train, y_train, X_val, y_val)


class TestTrainerEarlyStoppingNew:  # ← NOWA KLASA TESTÓW
    """Test early stopping functionality."""

    def test_early_stopping_triggers(self, simple_model, sample_data):
        """Test that early stopping triggers when no improvement."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = Trainer(model=simple_model, epochs=100, batch_size=5, patience=5)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # Early stopping should trigger before 100 epochs
        assert history["total_epochs_trained"] < 100
        assert history["stopped_early"] is True

    def test_early_stopping_disabled_without_validation(self, simple_model, sample_data):
        """Test that early stopping is disabled without validation set."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=10, batch_size=5, patience=5)

        history = trainer.fit(X_train, y_train)

        # Should complete all epochs (no early stopping)
        assert history["total_epochs_trained"] == 10
        assert history["stopped_early"] is False

    def test_early_stopping_disabled_with_patience_none(self, simple_model, sample_data):
        """Test that early stopping is disabled when patience=None."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = Trainer(model=simple_model, epochs=10, batch_size=5, patience=None)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # Should complete all epochs
        assert history["total_epochs_trained"] == 10
        assert history["stopped_early"] is False

    def test_best_weights_restored(self, simple_model, sample_data):
        """Test that best weights are restored after early stopping."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = Trainer(model=simple_model, epochs=100, batch_size=5, patience=5)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # Evaluate with restored weights
        final_val_metrics = trainer.evaluate(X_val, y_val)

        # Final validation accuracy should match best recorded accuracy
        assert abs(final_val_metrics["accuracy"] - history["best_val_accuracy"]) < 0.01


class TestTrainerEvaluate:
    """Test trainer evaluate method."""

    def test_evaluate_returns_metrics_dict(self, simple_model, sample_data):
        """Test that evaluate returns metrics dictionary."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        metrics = trainer.evaluate(X_train, y_train)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_evaluate_metrics_are_floats(self, simple_model, sample_data):
        """Test that metrics are float values."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        metrics = trainer.evaluate(X_train, y_train)

        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_accuracy_in_valid_range(self, simple_model, sample_data):
        """Test that accuracy is in valid range [0, 1]."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        metrics = trainer.evaluate(X_train, y_train)

        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_after_training(self, simple_model, sample_data):
        """Test evaluate after training."""
        X_train, y_train, X_val, y_val = sample_data
        trainer = Trainer(model=simple_model, epochs=10, batch_size=5, patience=10)

        trainer.fit(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_val, y_val)

        assert "loss" in metrics
        assert "accuracy" in metrics


class TestTrainerPredict:
    """Test trainer predict method."""

    def test_predict_returns_array(self, simple_model, sample_data):
        """Test that predict returns numpy array."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)
        trainer.fit(X_train, y_train)

        X_test = np.random.randn(5, 4)
        predictions = trainer.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (5,)

    def test_predict_returns_valid_classes(self, simple_model, sample_data):
        """Test that predictions are valid class indices."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)
        trainer.fit(X_train, y_train)

        X_test = np.random.randn(10, 4)
        predictions = trainer.predict(X_test)

        assert all(0 <= p < 3 for p in predictions)

    def test_predict_delegates_to_model(self, simple_model, sample_data):
        """Test that predict delegates to underlying model."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)
        trainer.fit(X_train, y_train)

        X_test = np.random.randn(5, 4)
        trainer_predictions = trainer.predict(X_test)
        model_predictions = simple_model.predict(X_test)

        assert np.array_equal(trainer_predictions, model_predictions)


class TestTrainerUtilities:
    """Test trainer utility methods."""

    def test_get_model(self, simple_model):
        """Test getting underlying model."""
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=10)

        assert trainer.get_model() is simple_model

    def test_get_history_before_training(self, simple_model):
        """Test getting history before training."""
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=10)

        assert trainer.get_history() is None

    def test_get_history_after_training(self, simple_model, sample_data):
        """Test getting history after training."""
        X_train, y_train, _, _ = sample_data
        trainer = Trainer(model=simple_model, epochs=5, batch_size=5, patience=None)

        trainer.fit(X_train, y_train)
        history = trainer.get_history()

        assert history is not None
        assert "train_loss" in history

    def test_repr(self, simple_model):
        """Test string representation."""
        trainer = Trainer(model=simple_model, epochs=50, batch_size=16, patience=10)

        repr_str = repr(trainer)

        assert "Trainer" in repr_str
        assert "MLP" in repr_str
        assert "50" in repr_str
        assert "16" in repr_str


class TestTrainerIntegration:
    """Integration tests."""

    def test_full_training_pipeline(self, simple_model, sample_data):
        """Test complete training pipeline."""
        X_train, y_train, X_val, y_val = sample_data

        trainer = Trainer(model=simple_model, epochs=20, batch_size=5, patience=10, random_state=42)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        # May have less than 20 epochs due to early stopping
        assert len(history["train_loss"]) <= 20
        assert len(history["val_loss"]) <= 20

        metrics = trainer.evaluate(X_val, y_val)
        assert 0 <= metrics["accuracy"] <= 1

        predictions = trainer.predict(X_val)
        assert predictions.shape == (10,)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that same random state gives same results."""
        X_train, y_train, _, _ = sample_data

        model1 = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(learning_rate=0.01),
            random_state=42,
        )

        model2 = MLP(
            layer_sizes=[4, 8, 3],
            activations=[ReLU(), Softmax()],
            loss_function=CrossEntropyLoss(),
            optimizer=SGDMomentum(learning_rate=0.01),
            random_state=42,
        )

        trainer1 = Trainer(model=model1, epochs=10, batch_size=5, patience=None, random_state=42)
        trainer2 = Trainer(model=model2, epochs=10, batch_size=5, patience=None, random_state=42)

        history1 = trainer1.fit(X_train, y_train)
        history2 = trainer2.fit(X_train, y_train)

        assert np.allclose(history1["train_loss"], history2["train_loss"])
