import unittest

import numpy as np

import hnswlib


class SoftDeleteTestCase(unittest.TestCase):
    def testBatchMarkDeleted(self):
        dim = 8
        data = np.float32(np.random.random((20, dim)))
        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=20)
        p.add_items(data, np.arange(20))

        self.assertEqual(p.mark_deleted([1, 2, 3]), 3)
        self.assertEqual(p.mark_deleted([3, 4]), 1)  # 3 was already deleted
        self.assertEqual(sorted(p.get_ids_list()), [0] + list(range(5, 20)))
        labels, _ = p.knn_query(data[1:5], k=1)
        self.assertFalse(set(labels.flatten()) & {1, 2, 3, 4})

        self.assertEqual(p.unmark_deleted([1, 2, 3, 4]), 4)
        self.assertEqual(sorted(p.get_ids_list()), list(range(20)))

        with self.assertRaises(RuntimeError):
            p.mark_deleted([5, 100])

    def testBFDeleteVector(self):
        dim = 8
        data = np.float32(np.random.random((3, dim)))
        bf = hnswlib.BFIndex(space='l2', dim=dim)
        bf.init_index(max_elements=3)
        bf.add_items(data, [10, 11, 12])

        self.assertFalse(bf.delete_vector(42))
        self.assertTrue(bf.delete_vector(12))  # the last element
        self.assertTrue(bf.delete_vector(10))
        self.assertEqual(bf.get_current_count(), 1)

        bf.add_items(data[[2, 0]], [12, 10])
        labels, _ = bf.knn_query(data, k=1)
        self.assertEqual(labels.flatten().tolist(), [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
