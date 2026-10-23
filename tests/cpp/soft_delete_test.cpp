// Tests for our soft delete additions: batch markDelete/unmarkDelete on
// HierarchicalNSW and the fixed BruteforceSearch::removePoint.

#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void testBatchDelete() {
    int d = 4;
    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> index(&space, 16);
    std::vector<float> v(d, 0.0f);
    for (size_t i = 0; i < 8; i++) {
        v[0] = i;
        index.addPoint(v.data(), i);
    }

    // already deleted labels are skipped, the count is what actually changed
    assert(index.markDelete({1, 2, 3}) == 3);
    assert(index.markDelete({3, 4}) == 1);
    assert(index.getDeletedCount() == 4);
    assert(index.unmarkDelete({1, 2, 3, 4}) == 4);
    assert(index.getDeletedCount() == 0);

    bool thrown = false;
    try {
        index.markDelete({5, 100});
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    assert(thrown);
    assert(index.getDeletedCount() == 1);  // 5 was marked before 100 failed
}

void testBruteforceRemove() {
    int d = 4;
    hnswlib::L2Space space(d);
    hnswlib::BruteforceSearch<float> index(&space, 8);
    std::vector<float> v(d, 0.0f);
    for (size_t i = 0; i < 3; i++) {
        v[0] = i;
        index.addPoint(v.data(), i);
    }

    assert(!index.removePoint(42));
    assert(index.removePoint(2));  // the last element
    assert(index.removePoint(0));  // 1 moves into slot 0
    assert(index.cur_element_count == 1);

    // both can be added again and are found afterwards
    v[0] = 2;
    index.addPoint(v.data(), 2);
    v[0] = 0;
    index.addPoint(v.data(), 0);
    for (size_t i = 0; i < 3; i++) {
        v[0] = i;
        assert(index.searchKnn(v.data(), 1).top().second == i);
    }
}

}  // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    testBatchDelete();
    testBruteforceRemove();
    std::cout << "Test ok" << std::endl;
    return 0;
}
