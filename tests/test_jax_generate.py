"""Regression tests for the lightweight JAX generation entry point."""

import unittest
from unittest import mock

from magenta_rt.jax import generate


class GenerateMainTest(unittest.TestCase):

  @mock.patch('magenta_rt.__getattr__')
  def test_non_positive_duration_fails_before_loading_model(self, get_attribute):
    with self.assertRaisesRegex(ValueError, 'duration must be positive'):
      generate.main(duration=0)

    get_attribute.assert_not_called()
