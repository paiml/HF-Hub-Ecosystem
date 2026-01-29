"""Tests for trainer creation utilities."""

from unittest.mock import MagicMock, patch

from hf_ecosystem.training.trainer import (
    create_trainer,
    get_default_training_args,
)


class TestCreateTrainer:
    """Tests for create_trainer function."""

    @patch("hf_ecosystem.training.trainer.Trainer")
    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_create_trainer_returns_trainer(self, mock_args_class, mock_trainer_class):
        """create_trainer should return a Trainer instance."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        result = create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
        )

        assert result == mock_trainer
        mock_trainer_class.assert_called_once()

    @patch("hf_ecosystem.training.trainer.Trainer")
    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_create_trainer_with_eval_dataset(
        self, mock_args_class, mock_trainer_class
    ):
        """create_trainer should configure eval when eval_dataset provided."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args
        mock_trainer_class.return_value = MagicMock()

        create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
        )

        # Verify TrainingArguments were created with eval_strategy="epoch"
        args_kwargs = mock_args_class.call_args[1]
        assert args_kwargs["eval_strategy"] == "epoch"
        assert args_kwargs["load_best_model_at_end"] is True

    @patch("hf_ecosystem.training.trainer.Trainer")
    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_create_trainer_without_eval_dataset(
        self, mock_args_class, mock_trainer_class
    ):
        """create_trainer should disable eval when no eval_dataset."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args
        mock_trainer_class.return_value = MagicMock()

        create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            eval_dataset=None,
        )

        args_kwargs = mock_args_class.call_args[1]
        assert args_kwargs["eval_strategy"] == "no"
        assert args_kwargs["load_best_model_at_end"] is False

    @patch("hf_ecosystem.training.trainer.Trainer")
    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_create_trainer_custom_hyperparams(
        self, mock_args_class, mock_trainer_class
    ):
        """create_trainer should accept custom hyperparameters."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args
        mock_trainer_class.return_value = MagicMock()

        create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            output_dir="./custom_output",
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-4,
            weight_decay=0.05,
        )

        args_kwargs = mock_args_class.call_args[1]
        assert args_kwargs["output_dir"] == "./custom_output"
        assert args_kwargs["num_train_epochs"] == 5
        assert args_kwargs["per_device_train_batch_size"] == 16
        assert args_kwargs["learning_rate"] == 1e-4
        assert args_kwargs["weight_decay"] == 0.05

    @patch("hf_ecosystem.training.trainer.Trainer")
    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_create_trainer_with_compute_metrics(
        self, mock_args_class, mock_trainer_class
    ):
        """create_trainer should pass compute_metrics function."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        mock_compute_metrics = MagicMock()
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args
        mock_trainer_class.return_value = MagicMock()

        create_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            compute_metrics=mock_compute_metrics,
        )

        trainer_kwargs = mock_trainer_class.call_args[1]
        assert trainer_kwargs["compute_metrics"] == mock_compute_metrics


class TestGetDefaultTrainingArgs:
    """Tests for get_default_training_args function."""

    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_get_default_training_args_returns_args(self, mock_args_class):
        """get_default_training_args should return TrainingArguments."""
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args

        result = get_default_training_args()

        assert result == mock_args
        mock_args_class.assert_called_once()

    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_get_default_training_args_with_custom_values(self, mock_args_class):
        """get_default_training_args should accept custom output_dir and epochs."""
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args

        get_default_training_args(output_dir="./my_output", num_epochs=10)

        args_kwargs = mock_args_class.call_args[1]
        assert args_kwargs["output_dir"] == "./my_output"
        assert args_kwargs["num_train_epochs"] == 10

    @patch("hf_ecosystem.training.trainer.TrainingArguments")
    def test_get_default_training_args_has_sensible_defaults(self, mock_args_class):
        """get_default_training_args should use sensible default values."""
        mock_args = MagicMock()
        mock_args_class.return_value = mock_args

        get_default_training_args()

        args_kwargs = mock_args_class.call_args[1]
        assert args_kwargs["per_device_train_batch_size"] == 8
        assert args_kwargs["learning_rate"] == 2e-5
        assert args_kwargs["weight_decay"] == 0.01
        assert args_kwargs["warmup_ratio"] == 0.1
        assert args_kwargs["eval_strategy"] == "epoch"
        assert args_kwargs["save_strategy"] == "epoch"
        assert args_kwargs["load_best_model_at_end"] is True
