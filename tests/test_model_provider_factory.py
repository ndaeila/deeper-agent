"""Tests for ModelProviderFactory."""

from unittest.mock import MagicMock, patch

import pytest

from odr.integrations import ConfiguredProvider, ModelProviderFactory, get_factory


class TestModelProviderFactory:
    """Tests for the main factory class."""

    def test_configure_returns_configured_provider(self):
        """configure() should return a ConfiguredProvider instance."""
        factory = ModelProviderFactory()
        provider = factory.configure("openai", model="gpt-4o", temperature=0.7)

        assert isinstance(provider, ConfiguredProvider)
        assert provider.provider == "openai"
        assert provider.config["model"] == "gpt-4o"
        assert provider.config["temperature"] == 0.7

    def test_convenience_methods(self):
        """Convenience methods should configure correct providers."""
        factory = ModelProviderFactory()

        openai = factory.openai()
        assert openai.provider == "openai"
        assert openai.config["model"] == "gpt-4o"

        anthropic = factory.anthropic()
        assert anthropic.provider == "anthropic"
        assert anthropic.config["model"] == "claude-sonnet-4-20250514"

        google = factory.google()
        assert google.provider == "google"
        assert google.config["model"] == "gemini-2.0-flash"

        ollama = factory.ollama()
        assert ollama.provider == "ollama"
        assert ollama.config["model"] == "llama3.2"

    def test_convenience_methods_accept_overrides(self):
        """Convenience methods should accept config overrides."""
        factory = ModelProviderFactory()
        provider = factory.openai(model="gpt-4o-mini", temperature=0.5)

        assert provider.config["model"] == "gpt-4o-mini"
        assert provider.config["temperature"] == 0.5


class TestConfiguredProvider:
    """Tests for the ConfiguredProvider subfactory."""

    def test_immutability(self):
        """ConfiguredProvider should be immutable."""
        provider = ConfiguredProvider(provider="openai", config={"model": "gpt-4o"})

        with pytest.raises(AttributeError):
            provider.provider = "anthropic"

    def test_with_config_creates_new_instance(self):
        """with_config() should return a new provider with merged config."""
        original = ConfiguredProvider(provider="openai", config={"model": "gpt-4o"})
        updated = original.with_config(temperature=0.5)

        # Original unchanged
        assert "temperature" not in original.config

        # New instance has merged config
        assert updated.config["model"] == "gpt-4o"
        assert updated.config["temperature"] == 0.5

    def test_with_config_overrides_existing(self):
        """with_config() should override existing config values."""
        original = ConfiguredProvider(
            provider="openai", config={"model": "gpt-4o", "temperature": 0.7}
        )
        updated = original.with_config(temperature=0.2)

        assert original.config["temperature"] == 0.7
        assert updated.config["temperature"] == 0.2

    @patch("odr.integrations.ModelProviderFactory.ChatOpenAI")
    def test_create_calls_langchain_model(self, mock_chat_openai):
        """create() should instantiate the correct LangChain model."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        provider = ConfiguredProvider(
            provider="openai", config={"model": "gpt-4o", "temperature": 0.7}
        )
        result = provider.create()

        mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.7)
        assert result is mock_instance

    @patch("odr.integrations.ModelProviderFactory.ChatOpenAI")
    def test_create_with_overrides(self, mock_chat_openai):
        """create() should merge overrides with base config."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        provider = ConfiguredProvider(
            provider="openai", config={"model": "gpt-4o", "temperature": 0.7}
        )
        provider.create(temperature=0.2, max_tokens=100)

        mock_chat_openai.assert_called_once_with(
            model="gpt-4o", temperature=0.2, max_tokens=100
        )


class TestProviderCreation:
    """Tests for model creation with different providers."""

    @patch("odr.integrations.ModelProviderFactory.ChatOpenAI")
    def test_openai_creation(self, mock_model):
        """Should create OpenAI model correctly."""
        factory = ModelProviderFactory()
        provider = factory.openai()
        provider.create()

        mock_model.assert_called_once()

    @patch("odr.integrations.ModelProviderFactory.ChatAnthropic")
    def test_anthropic_creation(self, mock_model):
        """Should create Anthropic model correctly."""
        factory = ModelProviderFactory()
        provider = factory.anthropic()
        provider.create()

        mock_model.assert_called_once()

    @patch("odr.integrations.ModelProviderFactory.ChatGoogleGenerativeAI")
    def test_google_creation(self, mock_model):
        """Should create Google model correctly."""
        factory = ModelProviderFactory()
        provider = factory.google()
        provider.create()

        mock_model.assert_called_once()

    @patch("odr.integrations.ModelProviderFactory.ChatOllama")
    def test_ollama_creation(self, mock_model):
        """Should create Ollama model correctly."""
        factory = ModelProviderFactory()
        provider = factory.ollama()
        provider.create()

        mock_model.assert_called_once()

    def test_unknown_provider_raises_error(self):
        """Should raise ValueError for unknown providers."""
        factory = ModelProviderFactory()
        provider = factory.configure("unknown_provider")

        with pytest.raises(ValueError, match="Unknown provider"):
            provider.create()


class TestSingleton:
    """Tests for the module-level singleton."""

    def test_get_factory_returns_same_instance(self):
        """get_factory() should return the same instance."""
        factory1 = get_factory()
        factory2 = get_factory()

        assert factory1 is factory2

    def test_get_factory_returns_model_provider_factory(self):
        """get_factory() should return a ModelProviderFactory."""
        factory = get_factory()
        assert isinstance(factory, ModelProviderFactory)
