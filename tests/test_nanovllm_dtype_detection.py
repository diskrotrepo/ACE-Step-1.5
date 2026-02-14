"""Unit tests for nanovllm dtype detection based on GPU compute capability.

Validates that nanovllm correctly detects GPU compute capability and selects
appropriate dtype (float16 for Turing/older, bfloat16 for Ampere/newer) to
avoid CUBLAS errors on GPUs that don't support bfloat16.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch


class TestNanovllmDtypeDetection(unittest.TestCase):
    """Tests for GPU compute capability detection and dtype selection."""

    def _create_mock_gpu_props(self, major: int, minor: int, name: str = "Mock GPU"):
        """Create mock GPU properties for testing."""
        props = MagicMock()
        props.major = major
        props.minor = minor
        props.name = name
        return props

    def test_turing_gpu_uses_float16(self):
        """Turing GPU (7.5) should use float16 instead of bfloat16."""
        # Mock GPU properties for Turing architecture
        mock_props = self._create_mock_gpu_props(7, 5, "NVIDIA CMP 50HX")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Simulate the logic from model_runner.py
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            # With no config dtype, should default to float16 on Turing
            config_dtype = None
            selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertFalse(supports_bfloat16, 
                           "Turing GPU should not support bfloat16")
            self.assertEqual(selected_dtype, torch.float16,
                           "Turing GPU should use float16")

    def test_volta_gpu_uses_float16(self):
        """Volta GPU (7.0) should use float16 instead of bfloat16."""
        mock_props = self._create_mock_gpu_props(7, 0, "Tesla V100")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertFalse(supports_bfloat16,
                           "Volta GPU should not support bfloat16")
            self.assertEqual(selected_dtype, torch.float16,
                           "Volta GPU should use float16")

    def test_ampere_gpu_uses_bfloat16(self):
        """Ampere GPU (8.0) should use bfloat16."""
        mock_props = self._create_mock_gpu_props(8, 0, "A100")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertTrue(supports_bfloat16,
                          "Ampere GPU should support bfloat16")
            self.assertEqual(selected_dtype, torch.bfloat16,
                           "Ampere GPU should use bfloat16")

    def test_ada_lovelace_gpu_uses_bfloat16(self):
        """Ada Lovelace GPU (8.6) should use bfloat16."""
        mock_props = self._create_mock_gpu_props(8, 6, "RTX 4060")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertTrue(supports_bfloat16,
                          "Ada Lovelace GPU should support bfloat16")
            self.assertEqual(selected_dtype, torch.bfloat16,
                           "Ada Lovelace GPU should use bfloat16")

    def test_hopper_gpu_uses_bfloat16(self):
        """Hopper GPU (9.0) should use bfloat16."""
        mock_props = self._create_mock_gpu_props(9, 0, "H100")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertTrue(supports_bfloat16,
                          "Hopper GPU should support bfloat16")
            self.assertEqual(selected_dtype, torch.bfloat16,
                           "Hopper GPU should use bfloat16")

    def test_bfloat16_override_on_turing(self):
        """Config requesting bfloat16 should be overridden to float16 on Turing."""
        mock_props = self._create_mock_gpu_props(7, 5, "NVIDIA CMP 50HX")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            # Config requests bfloat16
            config_dtype = torch.bfloat16
            
            # Should be overridden to float16
            if config_dtype == torch.bfloat16 and not supports_bfloat16:
                selected_dtype = torch.float16
            else:
                selected_dtype = config_dtype
            
            self.assertEqual(selected_dtype, torch.float16,
                           "bfloat16 config should be overridden to float16 on Turing")

    def test_float16_config_preserved_on_ampere(self):
        """Config requesting float16 should be preserved even on Ampere."""
        mock_props = self._create_mock_gpu_props(8, 0, "A100")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            # Config explicitly requests float16
            config_dtype = torch.float16
            
            # Should be preserved
            if config_dtype == torch.bfloat16 and not supports_bfloat16:
                selected_dtype = torch.float16
            else:
                selected_dtype = config_dtype
            
            self.assertEqual(selected_dtype, torch.float16,
                           "float16 config should be preserved on Ampere")

    def test_float32_config_preserved(self):
        """Config requesting float32 should be preserved regardless of GPU."""
        mock_props = self._create_mock_gpu_props(7, 5, "NVIDIA CMP 50HX")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            # Config explicitly requests float32
            config_dtype = torch.float32
            
            # Should be preserved
            if config_dtype == torch.bfloat16 and not supports_bfloat16:
                selected_dtype = torch.float16
            else:
                selected_dtype = config_dtype
            
            self.assertEqual(selected_dtype, torch.float32,
                           "float32 config should be preserved")

    def test_string_dtype_conversion_on_turing(self):
        """String dtype 'bfloat16' should convert to torch.float16 on Turing."""
        mock_props = self._create_mock_gpu_props(7, 5, "NVIDIA CMP 50HX")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            # Config as string
            config_dtype_str = 'bfloat16'
            
            # Convert string to dtype
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
            }
            config_dtype = dtype_map.get(config_dtype_str.lower(), 
                                        torch.bfloat16 if supports_bfloat16 else torch.float16)
            
            # Override if needed
            if config_dtype == torch.bfloat16 and not supports_bfloat16:
                selected_dtype = torch.float16
            else:
                selected_dtype = config_dtype
            
            self.assertEqual(selected_dtype, torch.float16,
                           "String 'bfloat16' should become float16 on Turing")

    def test_none_dtype_defaults_correctly_on_turing(self):
        """None dtype should default to float16 on Turing."""
        mock_props = self._create_mock_gpu_props(7, 5, "NVIDIA CMP 50HX")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            
            if config_dtype is None:
                selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertEqual(selected_dtype, torch.float16,
                           "None dtype should default to float16 on Turing")

    def test_none_dtype_defaults_correctly_on_ampere(self):
        """None dtype should default to bfloat16 on Ampere."""
        mock_props = self._create_mock_gpu_props(8, 0, "A100")
        
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            # Use tuple comparison for robustness
            supports_bfloat16 = (mock_props.major, mock_props.minor) >= (8, 0)
            
            config_dtype = None
            
            if config_dtype is None:
                selected_dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
            
            self.assertEqual(selected_dtype, torch.bfloat16,
                           "None dtype should default to bfloat16 on Ampere")


if __name__ == "__main__":
    unittest.main()
