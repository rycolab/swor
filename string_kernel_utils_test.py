###############################################################################################
# Tests for string kernel
###############################################################################################

from string_kernel_utils import dynamic_programming_substring_kernel_k
from string_kernel_utils import substring_kernel_k, normalized_substring_kernel_k, test_substring_kernel_k
from utils import dynamic_programming_substring_kernel_k_efficient, lodhi_normalization

p = 3
decay = 0.5
s = [1, 2, 3, 5, 10, 5, 7, 12]
t = [1, 3, 7, 10, 2, 7, 20, 20, 1]

# symmetry
assert(dynamic_programming_substring_kernel_k(s, t, p, decay) == dynamic_programming_substring_kernel_k(t, s, p, decay))
assert(dynamic_programming_substring_kernel_k_efficient([1, 2], [1, 3], p, decay) ==
       dynamic_programming_substring_kernel_k_efficient([1, 3], [1, 2], p, decay))
assert(substring_kernel_k(s, t, p, decay) == substring_kernel_k(t, s, p, decay))
assert(normalized_substring_kernel_k(s, t, p, decay) == normalized_substring_kernel_k(t, s, p, decay))
assert(test_substring_kernel_k("abcde", "abdef", p, decay) == test_substring_kernel_k("abdef", "abcde", p, decay))

# equality
assert(dynamic_programming_substring_kernel_k([1, 3], [1, 4], 3, decay) ==
       dynamic_programming_substring_kernel_k_efficient([1, 3], [1, 4], 3, decay))
assert(dynamic_programming_substring_kernel_k(s, t, p, decay)[p] ==
       substring_kernel_k(s, t, p, decay))
assert(substring_kernel_k("abcde", "abdef", p, decay) == test_substring_kernel_k("abcde", "abdef", p, decay))

# normalization
assert(normalized_substring_kernel_k([1, 2], [1, 2], 2, decay) == 1 ==
       lodhi_normalization(dynamic_programming_substring_kernel_k([1,2], [1,2], 2, decay), [1,2], [1,2], 2, decay)[2])

# base case
assert(dynamic_programming_substring_kernel_k([1, 3], [1,2,3,4,5], 4, decay)[4] == 0)
assert(dynamic_programming_substring_kernel_k([], [], p, decay)[p] == 0)


assert(dynamic_programming_substring_kernel_k([1,2,3,5], [1,2,3,4,5], 3, decay)[3] >
       dynamic_programming_substring_kernel_k([1,7,8,9], [1,2,3,4,5], 3, decay)[3])


