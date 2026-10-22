"""CPU budget respects affinity and reports Windows processor-group limits."""
import unittest
from unittest.mock import patch,MagicMock
from deta_generator3.resources import cpu_inventory


class ResourceTests(unittest.TestCase):
    def inventory(self,logical,affinity,windows):
        proc = MagicMock()
        proc.cpu_affinity.return_value = list(range(affinity))
        with patch('deta_generator3.resources.psutil.cpu_count',return_value=logical), \
             patch('deta_generator3.resources.psutil.Process',return_value=proc), \
             patch('deta_generator3.resources._windows_cpu_info',return_value=windows):
            return cpu_inventory()

    def test_affinity_restriction_on_single_group(self):
        self.assertEqual(self.inventory(64,16,{'total':64,'groups':1,'cross_group_default':True})['budget_cpus'],16)

    def test_modern_windows_does_not_mistake_primary_group_for_whole_machine(self):
        inventory = self.inventory(64,64,{'total':128,'groups':2,'cross_group_default':True})
        self.assertEqual(inventory['budget_cpus'],128)
        self.assertEqual(inventory['windows_total_logical_cpus'],128)

    def test_older_windows_keeps_safe_budget_and_exposes_total(self):
        inventory = self.inventory(128,64,{'total':128,'groups':2,'cross_group_default':False})
        self.assertEqual(inventory['budget_cpus'],64)
        self.assertTrue(inventory['notes'])

    def test_restricted_affinity_on_multi_group_is_respected(self):
        self.assertEqual(self.inventory(128,4,{'total':128,'groups':2,'cross_group_default':True})['budget_cpus'],4)


if __name__ == '__main__': unittest.main()
